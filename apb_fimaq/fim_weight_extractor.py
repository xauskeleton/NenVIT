"""
Extract diagonal FIM at WEIGHT level for APB partition.

Bridge formula (paper-consistent diagonal approximation):
    For Linear z = W @ x + b,  ∂z_i / ∂W_ij = x_j
    Therefore:
        F_diag(W_ij) ≈ F_diag(z_i) · E[x_j²]

Where:
  - F_diag(z) is computed by FIMA-Q via Theorem 3.2:
        F_diag(z_i) = ∇L_KL(z_i) / Δz_i      (eq 11 in paper)
    In practice we approximate with E[|∇L_KL|] (averaged over calib samples).
  - E[x_j²] is the mean of squared input activations over the calibration batch.

This module hooks the model during a forward+backward KL pass and outputs
a dict {layer_name: fim_W_tensor} matching each Linear's weight shape.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class FIMWeightExtractor:
    def __init__(self, model, target_modules: dict, calib_loader, temperature: float = 20.0):
        """
        target_modules: {name: nn.Linear-subclass} — the Linears we want F_diag(W) for.
        """
        self.model = model
        self.targets = target_modules
        self.calib_loader = calib_loader
        self.temperature = temperature
        self.x_sq_accum = {name: None for name in target_modules}     # E[x²] per layer
        self.grad_z_accum = {name: None for name in target_modules}   # E[|∇L_KL(z)|] per layer
        self.n_samples = 0
        self._hooks = []

    # ------------------------------------------------------------------
    def _install_hooks(self):
        def make_fwd_hook(name):
            def hook(module, inputs, output):
                x = inputs[0].detach()
                # x shape: (..., in_features). Reduce over all dims except last → per-channel E[x²]
                x_sq = (x.float() ** 2).reshape(-1, x.shape[-1]).mean(dim=0)  # (in_features,)
                if self.x_sq_accum[name] is None:
                    self.x_sq_accum[name] = x_sq * x.shape[0]
                else:
                    self.x_sq_accum[name] += x_sq * x.shape[0]
            return hook

        def make_bwd_hook(name):
            def hook(module, grad_input, grad_output):
                # grad of the LINEAR's OUTPUT w.r.t. KL loss
                g = grad_output[0].detach()
                # Aggregate per out_feature: E[|∇z_i|]
                g_abs = g.float().abs().reshape(-1, g.shape[-1]).mean(dim=0)  # (out_features,)
                if self.grad_z_accum[name] is None:
                    self.grad_z_accum[name] = g_abs * g.shape[0]
                else:
                    self.grad_z_accum[name] += g_abs * g.shape[0]
            return hook

        for name, mod in self.targets.items():
            self._hooks.append(mod.register_forward_hook(make_fwd_hook(name)))
            self._hooks.append(mod.register_full_backward_hook(make_bwd_hook(name)))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    def run(self, device='cuda'):
        """Run forward+backward over calib_loader with KL loss; collect stats."""
        logging.info(f'FIMWeightExtractor: collecting stats over {len(self.calib_loader)} batches '
                     f'for {len(self.targets)} target layers')

        # Save current modes so we can restore at the end
        saved_modes = {id(m): m.mode for m in self.model.modules() if hasattr(m, 'mode')}

        # First pass: get FP softmax targets — FORCE all modules to 'raw' mode
        # (matches FIMA-Q's BlockReconstructor.init_block_raw_inp_outp logic)
        self._set_all_modes('raw')
        raw_softmax = []
        self.model.eval()
        with torch.no_grad():
            for inp, _ in self.calib_loader:
                inp = inp.to(device)
                logits = self.model(inp) / self.temperature
                raw_softmax.append(F.softmax(logits, dim=-1).detach())

        # Second pass: switch to quant_forward so backward measures quant-error gradient
        self._set_all_modes('quant_forward')
        # CRITICAL: enable training_mode on quantizers so they use round_ste (differentiable
        # via STE) instead of torch.round (non-diff). Without this, gradients are blocked
        # at every a_quantizer/w_quantizer and can't reach target Linears via the residual
        # branch.
        saved_training = []
        for m in self.model.modules():
            if hasattr(m, 'training_mode'):
                saved_training.append((m, m.training_mode))
                m.init_training()
        self._install_hooks()
        self.n_samples = 0
        for i, (inp, _) in enumerate(self.calib_loader):
            self.model.zero_grad(set_to_none=True)
            inp = inp.to(device)
            pred = self.model(inp) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_softmax[i], reduction='batchmean')
            loss.backward()
            self.n_samples += inp.size(0)
        self._remove_hooks()
        self.model.zero_grad(set_to_none=True)

        # Normalize accumulators
        for name in self.targets:
            if self.x_sq_accum[name] is not None:
                self.x_sq_accum[name] /= self.n_samples
            if self.grad_z_accum[name] is not None:
                self.grad_z_accum[name] /= self.n_samples
        logging.info(f'  collected over {self.n_samples} samples')

        # Restore training_mode (quantizers go back to torch.round path)
        for m, was in saved_training:
            if not was:
                m.end_training()

        # Restore module modes
        for m in self.model.modules():
            if hasattr(m, 'mode') and id(m) in saved_modes:
                m.mode = saved_modes[id(m)]

    # ------------------------------------------------------------------
    def _set_all_modes(self, mode):
        for m in self.model.modules():
            if hasattr(m, 'mode'):
                m.mode = mode

    # ------------------------------------------------------------------
    def fim_weight(self, name: str) -> torch.Tensor:
        """
        Build F_diag(W) for layer `name`:
            F_diag(W_ij) ≈ E[|∇z_i|] · E[x_j²]
        Returns tensor with same shape as module.weight: (out_features, in_features).
        For qkv (n_V=3 if applicable), FIM has same shape — splitting per Q/K/V
        happens later in APBWeightQuantizer via n_V reshape.
        """
        mod = self.targets[name]
        x_sq = self.x_sq_accum[name]      # (in_features,)
        g_abs = self.grad_z_accum[name]   # (out_features,)
        if x_sq is None or g_abs is None:
            raise RuntimeError(f'No stats collected for {name}; did run() execute?')
        # Outer product (out, in)
        fim = g_abs.unsqueeze(1) * x_sq.unsqueeze(0)
        # Match weight shape exactly
        expected = mod.weight.shape
        if fim.shape != expected:
            raise RuntimeError(f'shape mismatch for {name}: fim={fim.shape}, weight={expected}')
        return fim.detach().cpu()

    def all_fim(self) -> dict:
        return {name: self.fim_weight(name) for name in self.targets}
