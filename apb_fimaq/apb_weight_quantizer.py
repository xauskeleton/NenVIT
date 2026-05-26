"""
APB Weight Quantizer — drop-in replacement for FIMA-Q's AdaRoundQuantizer.

Operator (paper APB, eq 3):
    APB(w) = α · sign(w)   if mask[i,j] == True   (binary zone, 1-bit)
             w             if mask[i,j] == False  (FP zone, 32-bit)

Backward: STE with g(w) = w  →  ∂APB/∂w = 1 everywhere.

In PTQ setting we keep α (and δ for reference) FIXED from init:
    α = mean(|w|),  δ = 3 · std(w)
Mask is set EXTERNALLY by an APB-partition step that uses FIM importance,
not by the magnitude rule of the original APB paper.

For QKV layers, FIMA-Q reshapes weight to (n_V, crb_rows, in_features) so
Q/K/V can have independent scales. APBWeightQuantizer preserves this by
keeping a separate α per n_V group.
"""
import torch
import torch.nn as nn


class APBWeightQuantizer(nn.Module):
    def __init__(self, weight_tensor: torch.Tensor, n_V: int = 1):
        """
        weight_tensor: (n_V, crb_rows, in_features) — same shape FIMA-Q feeds AdaRound.
                       For non-qkv layers n_V=1, so it's effectively (1, out, in).
        """
        super().__init__()
        assert weight_tensor.dim() == 3 and weight_tensor.shape[0] == n_V, (
            f'expected (n_V={n_V}, crb_rows, in_features), got {tuple(weight_tensor.shape)}')
        self.n_V = n_V

        with torch.no_grad():
            flat = weight_tensor.reshape(n_V, -1)
            apb_alpha = flat.abs().mean(dim=1).clone()     # (n_V,)
            apb_delta = 3.0 * flat.std(dim=1).clone()      # (n_V,)
        # Fixed (not Parameters) in PTQ setting
        self.register_buffer('apb_alpha', apb_alpha)
        self.register_buffer('apb_delta', apb_delta)

        # Binary mask: True = binarize to ±α, False = keep FP.
        # Set externally via .set_mask() after FIM-based partition.
        self.register_buffer('mask', torch.zeros_like(weight_tensor, dtype=torch.bool))

        # --- Compatibility shims with AdaRoundQuantizer interface ---
        # FIMA-Q's reconstruct_single_block does:
        #     w_params += [module.w_quantizer.alpha]
        # We provide a dummy frozen Parameter so Adam doesn't crash. (Adam ignores
        # params with requires_grad=False AND no gradient — but we keep grad off
        # to ensure APB layers contribute 0 trainable weight params.)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Used by AdaRound's round_loss regularizer. We return a constant 0.5 from
        # get_soft_targets() so the regularization contributes 0 for APB layers.
        self.soft_targets = False
        self.n_bits = 1
        self.inited = True

    # ------------------------------------------------------------------
    # Mask management (called by Phase-C APB partition step)
    # ------------------------------------------------------------------
    def set_mask(self, mask: torch.Tensor):
        """mask: bool tensor same shape as weight_tensor passed in __init__."""
        assert mask.shape == self.mask.shape, (
            f'mask shape {tuple(mask.shape)} != expected {tuple(self.mask.shape)}')
        self.mask.data.copy_(mask.bool())

    def set_mask_from_fim(self, fim_diag: torch.Tensor, binary_ratio: float):
        """
        Build mask by thresholding FIM importance:
            mask[i,j] = (fim_diag[i,j] < τ)   where τ = percentile(fim_diag, binary_ratio)
        Low FIM → less important → binarize.
        """
        assert fim_diag.shape == self.mask.shape
        flat = fim_diag.float().flatten()
        tau = torch.quantile(flat, binary_ratio)
        self.mask.data.copy_(fim_diag < tau)

    # ------------------------------------------------------------------
    # Forward / hard rounding
    # ------------------------------------------------------------------
    def _alpha_broadcast(self):
        # (n_V,) → (n_V, 1, 1) for broadcast over (n_V, crb_rows, in_features)
        return self.apb_alpha.view(self.n_V, 1, 1)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        APB(w) with STE.
        w shape: (n_V, crb_rows, in_features). Returns same shape.
        """
        alpha = self._alpha_broadcast()
        w_bin = alpha * torch.sign(w)
        w_apb = torch.where(self.mask, w_bin, w)
        # STE: forward = w_apb, backward = identity through w
        return w + (w_apb - w).detach()

    def get_hard_value(self, w: torch.Tensor) -> torch.Tensor:
        """Final (no-STE) APB output. Used at end of reconstruction to bake into module.weight."""
        with torch.no_grad():
            alpha = self._alpha_broadcast()
            return torch.where(self.mask, alpha * torch.sign(w), w)

    # ------------------------------------------------------------------
    # AdaRound-API compatibility (so FIMA-Q's existing code paths work)
    # ------------------------------------------------------------------
    def get_soft_targets(self):
        """AdaRound's round_loss regularizer calls this. Return 0.5 so loss term = 0."""
        return torch.full_like(self.mask, 0.5, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def binary_ratio(self) -> float:
        return self.mask.float().mean().item()

    def storage_bits_per_weight(self) -> float:
        """Effective bits/weight: 1·p + 32·(1-p) where p = binary ratio."""
        p = self.binary_ratio
        return 1.0 * p + 32.0 * (1.0 - p)

    def __repr__(self):
        alpha_str = ', '.join(f'{a:.4f}' for a in self.apb_alpha.tolist())
        return (f'APBWeightQuantizer(n_V={self.n_V}, '
                f'α=[{alpha_str}], binary_ratio={self.binary_ratio:.3f}, '
                f'eff_bits={self.storage_bits_per_weight():.2f})')
