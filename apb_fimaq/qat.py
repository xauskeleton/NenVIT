"""
APB QAT on Swin / ViT (DeiT) — DPLR-FIM-based partition + end-to-end fine-tuning.
Model chosen with --model (default swin_small_patch4_window7_224).

Importance ranking uses FIMA-Q's DPLR-FIM (Diagonal + Low-Rank, paper Eq 20-21):
    F_DPLR(W_ij) = (p1 · E[(∇z_i)²]  +  p2 · E[|∇z_i|]) · E[x_j²]
                    ↑ rank-k (L2)       ↑ diag (L1)      ↑ activation
    Mask = (F_DPLR < τ percentile)

Flow:
  1. Load pretrained model (Swin/ViT/DeiT via --model)
  2. Compute DPLR-FIM (k=5 gradient samples by default, ~5 forward+backward)
  3. For each target Linear (96 of them, skip head/downsample/patch_embed):
        mask = (F_DPLR(W) < τ)
        wrap with APBLinear (binary zone: α·sign(w), FP zone: latent_weight)
  4. QAT: train end-to-end with CE loss on Tiny ImageNet
     - latent_weight + α learnable; mask FROZEN
     - Freeze α at half epochs (paper APB convention)
  5. Save best model

USAGE
=====
    python qat.py                              # default DPLR, br=0.75, 10 ep
    python qat.py --binary-ratio 0.85          # more aggressive
    python qat.py --fim-mode diag              # ablation: diag-only (faster, less accurate)
    python qat.py --fim-mode rank              # ablation: rank-k only
    python qat.py --fim-p1 1.5 --fim-p2 0.5    # tune DPLR weights
    python qat.py --debug                      # 1 epoch, k=1, smoke test

    # Quantize a PRUNED model (prune -> finetune -> APB-QAT). --init-model loads the
    # whole saved model object, so its pruned (heads/MLP) architecture is used as-is:
    python qat.py --dataset cifar100 \
        --init-model ckpt/pruned_cifar100/best_pruned_model.pt \
        --binary-ratio 0.75 --epochs 10
"""
import argparse
import copy
import math
import sys
from tqdm.auto import tqdm
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
# Data loaders are imported lazily inside main() per --dataset, so that e.g. a
# CIFAR run does not require HuggingFace `datasets` (needed only by the tiny loader).

# Finetuned FP baseline. QAT initialises from this (if present & shape-compatible)
# instead of raw timm-pretrained weights, so quantization starts from a model that is
# already trained for the target dataset (e.g. the CIFAR head is no longer random).
DEFAULT_INIT_CKPT = PROJECT_ROOT / 'ckpt' / 'best_swin.pth'


def init_from_baseline(model, ckpt_path=None):
    """Load a finetuned FP baseline state_dict into `model` in-place if it exists and fits.

    ckpt_path overrides DEFAULT_INIT_CKPT (ckpt/best_swin.pth, the Swin-S baseline) — pass a
    per-architecture baseline (e.g. ckpt_deit/best.pth) so different model families keep
    their own baseline instead of clobbering Swin's.

    Returns 'loaded' on success, 'none' if no checkpoint file, or 'mismatch' if the
    saved weights are shape-incompatible (e.g. a Swin ckpt against a DeiT model, or
    cifar100/100-class ckpt against a tiny/1000-class run) — model keeps timm-pretrained.
    """
    ckpt = Path(ckpt_path) if ckpt_path else DEFAULT_INIT_CKPT
    if not ckpt.exists():
        return 'none'
    sd = torch.load(ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:   # tolerate a last.pth-style checkpoint
        sd = sd['model']
    try:
        model.load_state_dict(sd, strict=True)
        return 'loaded'
    except RuntimeError as e:
        print(f'!! {ckpt.name} incompatible with current model, keeping '
              f'timm-pretrained: {str(e).splitlines()[0]}')
        return 'mismatch'


class _Tee:
    """Mirror writes to a file in addition to the original stream.

    tqdm writes to stderr by default, so teeing stdout keeps the log file
    free of progress-bar carriage-return spam — only print()s land in it.
    """
    def __init__(self, stream, fp):
        self.stream = stream
        self.fp = fp
    def write(self, s):
        self.stream.write(s)
        self.fp.write(s)
        self.fp.flush()
    def flush(self):
        self.stream.flush()
        self.fp.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)


# ============================================================
# APB LAYER
# ============================================================
class _APBSTE(torch.autograd.Function):
    """Custom STE preserving BOTH latent_weight AND α gradients.

    Forward:
        out[i,j] = α · sign(w[i,j])  if mask[i,j]  (binary zone)
                 = w[i,j]            otherwise     (FP zone)

    Backward (per paper APB Eq 4 + 8):
        ∂out/∂w_ij = 1      ∀(i,j)              ← STE identity for latent_weight
        ∂out/∂α    = sign(w_ij) χ_B(i,j)         ← α gets grad only from binary zone

    NOTE: The naive STE `w + (eff - w).detach()` blocks α's gradient because
    the entire (eff - w) is treated as a constant. We need a custom Function
    to correctly propagate both gradients separately.
    """
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, w, alpha, mask):
        binary = alpha * torch.sign(w)
        out = torch.where(mask, binary, w)
        ctx.save_for_backward(w, mask)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_out):
        w, mask = ctx.saved_tensors
        grad_w = grad_out                                          # STE identity
        grad_alpha = (grad_out * torch.sign(w) * mask).sum()       # binary-only
        return grad_w, grad_alpha, None                            # mask: no grad


# ============================================================
# ACTIVATION QUANT — LSQ (Esser et al. 2020), learnable step + STE
# ============================================================
def _round_ste(x):
    """round() with a straight-through-estimator gradient (∂/∂x = 1)."""
    return (x.round() - x).detach() + x


def _grad_scale(x, scale):
    """Identity in forward; scales the gradient flowing into x by `scale`.
    LSQ trick to dampen the step-size gradient (g = 1/sqrt(N·Qp))."""
    return (x - x * scale).detach() + x * scale


class LSQActQuant(nn.Module):
    """Learned Step Size Quantization for activations (signed symmetric).

        x_q = round(clip(x / s, Qn, Qp)) · s,   Qn = -2^(b-1), Qp = 2^(b-1)-1

    One learnable step `s` per quantizer. The autograd-friendly
    _round_ste + _grad_scale form reproduces the exact LSQ gradient w.r.t. s
    (round(x/s) - x/s in-range, Qn/Qp out-of-range); gradient w.r.t. x is the
    STE identity inside the clip range and 0 outside. `s` is lazily initialised
    from the first batch (2·E|x|/√Qp), on train OR eval, via the `initialized`
    buffer — so the pre-QAT eval already uses a sane step.

    Signed symmetric (ViT activations after LN/GELU are two-sided). b>=32 → no-op.
    """
    def __init__(self, n_bits: int):
        super().__init__()
        self.n_bits = n_bits
        self.Qn = -(2 ** (n_bits - 1))
        self.Qp = 2 ** (n_bits - 1) - 1
        self.step = nn.Parameter(torch.ones(1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.uint8))

    @torch.no_grad()
    def _init_step(self, x):
        self.step.copy_(2 * x.detach().float().abs().mean() / math.sqrt(max(self.Qp, 1)))
        self.initialized.fill_(1)

    def forward(self, x):
        if self.n_bits >= 32:
            return x
        if self.initialized.item() == 0:
            self._init_step(x)
        s = _grad_scale(self.step, 1.0 / math.sqrt(max(x.numel() * self.Qp, 1)))
        return _round_ste(torch.clamp(x / s, self.Qn, self.Qp)) * s


# ============================================================
# BINARY ACTIVATION QUANT — 1-bit (XNOR/Bi-Real style)
# ============================================================
class _BinaryActSTE(torch.autograd.Function):
    """1-bit activation: out = scale · sign(x), levels {−scale, +scale}.

    Same scale·sign(·) form as APB weight binarization (_APBSTE), but for
    activations — no mask (binarize all), no stored latent (x is an input).

    Backward:
        ∂out/∂x     ≈ 1_{|x/scale| < 1}      ← Bi-Real/hardtanh STE (clip outside)
        ∂out/∂scale = sign(x)                 ← summed over all elements
    """
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x, scale):
        ctx.save_for_backward(x, scale)
        return scale * torch.sign(x)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_out):
        x, scale = ctx.saved_tensors
        # STE for x: pass gradient only where |x/scale| < 1 (clipped identity).
        grad_x = grad_out * (x.abs() < scale).to(grad_out.dtype)
        # scale gets ∂out/∂scale = sign(x), accumulated to its scalar shape.
        grad_scale = (grad_out * torch.sign(x)).sum().reshape(scale.shape)
        return grad_x, grad_scale


class BinaryActQuant(nn.Module):
    """1-bit activation quantizer: x_q = scale · sign(x), output ∈ {−scale, +scale}.

    Symmetric around 0 (unlike LSQActQuant, which at 1-bit degenerates to Qp=0 and
    kills the positive half). Learnable per-tensor `scale`, lazily initialised from
    the first batch to mean|x| (the L1-optimal binary magnitude, XNOR-Net Eq 6).
    Kept positive via abs() in forward, like APB's α. STE: Bi-Real clipped identity.
    """
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.uint8))

    @torch.no_grad()
    def _init_scale(self, x):
        self.scale.copy_(x.detach().float().abs().mean().clamp_min(1e-8))
        self.initialized.fill_(1)

    def forward(self, x):
        if self.initialized.item() == 0:
            self._init_scale(x)
        return _BinaryActSTE.apply(x, self.scale.abs())


def make_act_quant(act_bits: int):
    """Activation fake-quant factory: 0/≥32 → None (off), 1 → BinaryActQuant,
    2..31 → LSQActQuant. Centralises the 1-bit-is-special routing."""
    if not act_bits or act_bits >= 32:
        return None
    if act_bits == 1:
        return BinaryActQuant()
    return LSQActQuant(act_bits)


class APBLinear(nn.Module):
    """nn.Linear wrapper with APB partition.

    Forward (per weight w_ij):
        mask[i,j] = True   →  α · sign(w_ij)       (1-bit binary)
        mask[i,j] = False  →  w_ij                  (FP32, learnable)
    Backward: STE — both α and latent_weight receive correct gradients.
    """
    def __init__(self, linear: nn.Linear, mask: torch.Tensor, act_bits: int = 0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.latent_weight = nn.Parameter(linear.weight.data.clone())
        self.bias = linear.bias
        # α: scalar learnable, init from mean|w|; kept positive via abs() in forward
        self.alpha = nn.Parameter(self.latent_weight.detach().abs().mean())
        # Mask: True = binary. Frozen throughout training.
        self.register_buffer('mask', mask.bool())
        # Optional activation fake-quant on this layer's INPUT (0/32 = off).
        # act_bits==1 → BinaryActQuant (sign·scale); 2..31 → LSQActQuant.
        self.act_quant = make_act_quant(act_bits)

    def forward(self, x):
        if self.act_quant is not None:
            x = self.act_quant(x)
        # alpha.abs() to keep binary magnitude positive (paper convention).
        # Grad flows correctly through abs (sign(alpha) backward).
        eff_w = _APBSTE.apply(self.latent_weight, self.alpha.abs(), self.mask)
        return F.linear(x, eff_w, self.bias)

    @property
    def binary_ratio(self) -> float:
        return self.mask.float().mean().item()

    def storage_bits_per_weight(self, b_v: int = 32) -> float:
        """Avg bits/weight per APB paper Eq 10:
            Mem(W) = (n-s)·1 + s·(b_v + b_p) bits
        where:
            n   = layer size (out × in)
            s   = #FP entries = (1-p)·n
            b_v = FP value bit-width (32 default, matches paper)
            b_p = ⌈log₂(n-1)⌉ + 1 bits per FP position index
        """
        p = self.binary_ratio
        n = self.in_features * self.out_features
        b_p = math.ceil(math.log2(max(n - 1, 2))) + 1
        s = round((1 - p) * n)
        return ((n - s) + s * (b_v + b_p)) / n


# ============================================================
# TARGET LAYER FILTER
# ============================================================
SKIP_PATTERNS_SKIP = ('downsample.reduction', 'head', 'patch_embed')  # default: skip 4 critical
SKIP_PATTERNS_FULL = ('patch_embed',)                                  # full: only skip Conv2d


def get_target_layers(model: nn.Module, scope: str = 'skip') -> dict:
    """
    Return {name: module} of APB targets. Works for both pre-wrap (nn.Linear)
    and post-wrap (APBLinear) states.

    scope='skip' (default): 96 Linears — qkv/proj/fc1/fc2 × 24 blocks.
        Skips head.fc, 3 downsample.reduction, patch_embed (Conv2d anyway).
        Safe: doesn't touch classifier output or cross-resolution merge.

    scope='full': 100 Linears — adds head.fc + 3 downsample.reduction to APB.
        Aggressive: maximize compression coverage but risk accuracy drop.
        patch_embed.proj (Conv2d) still skipped since APBLinear only wraps nn.Linear.
    """
    if scope == 'skip':
        skip_patterns = SKIP_PATTERNS_SKIP
    elif scope == 'full':
        skip_patterns = SKIP_PATTERNS_FULL
    else:
        raise ValueError(f'scope must be "skip" or "full", got {scope!r}')

    targets = {}
    for n, m in model.named_modules():
        if isinstance(m, APBLinear):
            targets[n] = m
        elif isinstance(m, nn.Linear) and not any(p in n for p in skip_patterns):
            targets[n] = m
    return targets


# ============================================================
# FIM EXTRACTION — DPLR (Diagonal + Low-Rank), per FIMA-Q Eq (20-21)
# ============================================================
def compute_weight_dplr_fim(model, calib_loader, device,
                             n_batches=5, p1=1.0, p2=1.0, fim_mode='dplr',
                             scope='skip', full=False):
    """
    Compute DPLR-FIM-based importance for each target Linear's weights.

    Per FIMA-Q paper Section 3.3:
      F_DPLR(z) = α · F_rank-k(z) + (1-α) · F_diag(z)            (Eq 20)
      We use p1, p2 form: F_DPLR = p1·F_rank-k + p2·F_diag

    Estimators (computed from k=n_batches gradient samples per layer):
      L1 diagonal:  f_diag(i) = E_t[|∇z_i(t)|]    (Eq 11 simplified, paper code style)
      L2 rank-k:    f_rank(i) = E_t[(∇z_i(t))²]   (sample variance, diagonal of GᵀG/k)
      DPLR:         f_DPLR(i) = p1·f_rank(i) + p2·f_diag(i)

    Bridge to weight (Linear z = Wx, ∂z_a/∂W_ij = δ_ai·x_j):
      F_DPLR(W_ij) = f_DPLR(i) · E[x_j²]

    Args:
      n_batches: number of (forward + backward) calibration batches (ignored if full=True)
      full: if True, ignore n_batches and make ONE pass over the ENTIRE loader —
        exact importance with no calibration sampling error (esp. for 'fisher').
      p1: weight for rank-k (L2) component
      p2: weight for diag (L1) component
      fim_mode: 'dplr' | 'rank' | 'diag' | 'fisher' — switch for ablation
        'fisher' = exact empirical Fisher diagonal, per-token, WITHOUT the g⊥x
        factorization the others use: F(W_ij) = (1/T) Σ_t g_i(t)²·x_j(t)²,
        computed as matmul (g²)ᵀ(x²) so the per-token g–x correlation is kept
        (vs 'rank' = factorized E[g²]·E[x²]). Costs one (out,in) accumulator +
        caching x² between fwd/bwd → more memory.

    Returns:
      dict {name: F_DPLR(W) tensor of shape (out, in)}
    """
    targets = get_target_layers(model, scope=scope)
    fisher = (fim_mode == 'fisher')
    # Accumulators
    x_sq  = {n: 0.0 for n in targets}   # E[x²] per input channel
    g_abs = {n: 0.0 for n in targets}   # E[|g|]  per output channel  (for diag)
    g_sq  = {n: 0.0 for n in targets}   # E[g²]   per output channel  (for rank-k)
    gx_sq = {n: 0.0 for n in targets}   # exact Fisher: Σ_t g_i(t)²·x_j(t)²  (out,in)
    tok   = {n: 0 for n in targets}     # token count per layer (fisher mode)
    x_cache = {}                        # fisher: per-layer x² (T,in) between fwd & bwd
    n_samples = 0

    hooks = []
    for name, mod in targets.items():
        def make_fwd(n):
            def h(_m, inp, _out):
                x = inp[0].detach()
                B = x.shape[0]
                xf2 = (x.float() ** 2).reshape(-1, x.shape[-1])   # (T, in)
                if fisher:
                    x_cache[n] = xf2                              # keep per-token to pair with g
                else:
                    x_sq[n] = x_sq[n] + xf2.mean(dim=0) * B
            return h
        def make_bwd(n):
            def h(_m, _gi, go):
                g = go[0]
                if g is None: return
                B = g.shape[0]
                gf_flat = g.detach().float().reshape(-1, g.shape[-1])   # (T, out)
                if fisher:
                    xf2 = x_cache.pop(n, None)
                    if xf2 is None: return
                    gx_sq[n] = gx_sq[n] + (gf_flat ** 2).transpose(0, 1) @ xf2  # (out,in)
                    tok[n] = tok[n] + gf_flat.shape[0]
                else:
                    g_abs[n] = g_abs[n] + gf_flat.abs().mean(dim=0) * B
                    g_sq[n]  = g_sq[n]  + (gf_flat ** 2).mean(dim=0) * B
            return h
        hooks.append(mod.register_forward_hook(make_fwd(name)))
        hooks.append(mod.register_full_backward_hook(make_bwd(name)))

    model.train()
    crit = nn.CrossEntropyLoss()
    total = len(calib_loader) if full else n_batches
    for i, (x, y) in enumerate(tqdm(calib_loader, total=total, desc='FIM extract', leave=False)):
        if not full and i >= n_batches: break
        model.zero_grad()
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        n_samples += x.size(0)
        x_cache.clear()          # drop any un-consumed cached activations (fisher)
    for h in hooks: h.remove()
    model.zero_grad()

    fim = {}
    for name in targets:
        if fim_mode == 'fisher':
            if isinstance(gx_sq[name], float):       # layer never saw a gradient
                continue
            fim[name] = (gx_sq[name] / max(tok[name], 1)).cpu()  # (out,in) exact Fisher
            continue
        if isinstance(g_abs[name], float):
            continue
        gd = g_abs[name] / n_samples            # f_diag (L1)
        gr = g_sq[name]  / n_samples            # f_rank-k (L2)
        xs = x_sq[name]  / n_samples            # E[x²]

        if   fim_mode == 'diag': f_i = gd
        elif fim_mode == 'rank': f_i = gr
        elif fim_mode == 'dplr': f_i = p1 * gr + p2 * gd
        else: raise ValueError(f'fim_mode must be dplr|rank|diag|fisher, got {fim_mode}')

        fim[name] = (f_i.unsqueeze(1) * xs.unsqueeze(0)).cpu()  # (out, in)
    return fim


# Backwards-compat alias
compute_weight_fim = compute_weight_dplr_fim


# ============================================================
# APPLY APB (replace nn.Linear → APBLinear)
# ============================================================
def _partition_score(weight, fim, partition):
    """Per-weight importance for the APB partition. Convention: LOW score → binarize,
    HIGH score (top `1-binary_ratio`) → kept full-precision outlier.

      'fisher'    → empirical-Fisher importance (FIM-guided, our method; needs precomputed `fim`)
      'magnitude' → |w| (pure APB: keep the largest-magnitude weights FP)

    Returns a CPU float tensor shaped like the weight (out, in)."""
    if partition == 'fisher':
        if fim is None:
            raise ValueError("partition='fisher' needs a FIM dict for this layer")
        return fim.float()
    if partition == 'magnitude':
        return weight.detach().abs().float().cpu()
    raise ValueError(f'partition must be fisher|magnitude, got {partition!r}')


def apply_apb(model, fim_dict, binary_ratio, device, scope='skip', act_bits=0,
              partition='magnitude'):
    """Wrap each target Linear with APBLinear using a `partition`-percentile mask.
    partition: 'fisher' (FIM-guided) or 'magnitude' (pure APB, default).
    act_bits: LSQ activation bit-width on each wrapped layer's input (0/32 = off)."""
    targets = get_target_layers(model, scope=scope)
    for name, mod in targets.items():
        if isinstance(mod, APBLinear):
            continue  # already wrapped
        fim = fim_dict.get(name) if fim_dict else None
        score = _partition_score(mod.weight, fim, partition)
        tau = torch.quantile(score.flatten().float(), binary_ratio)
        mask = (score < tau)
        new_layer = APBLinear(mod, mask=mask, act_bits=act_bits).to(device)
        # Reassign in parent
        parent = model
        parts = name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_layer)
    return model


def update_masks_from_fim(model, fim_dict, binary_ratio, scope='skip', partition='magnitude'):
    """Refresh masks of APBLinear layers in-place using the chosen `partition` score.
    Keeps latent_weight, α, and structure intact — only mask buffer changes."""
    n_updated = 0
    n_flipped_total = 0
    n_weights_total = 0
    for name, mod in get_target_layers(model, scope=scope).items():
        if not isinstance(mod, APBLinear):
            continue
        if partition == 'fisher' and name not in fim_dict:
            continue
        fim = fim_dict.get(name) if fim_dict else None
        score = _partition_score(mod.latent_weight, fim, partition).to(mod.mask.device)
        tau = torch.quantile(score.flatten().float(), binary_ratio)
        new_mask = (score < tau)
        # Count how many positions changed
        flipped = (new_mask != mod.mask).sum().item()
        mod.mask.data.copy_(new_mask)
        n_updated += 1
        n_flipped_total += flipped
        n_weights_total += new_mask.numel()
    flip_pct = 100.0 * n_flipped_total / max(n_weights_total, 1)
    return n_updated, flip_pct


# ============================================================
# DPLR-FIM PER-BLOCK LOSS (distillation from FP teacher during QAT)
# Implements FIMA-Q paper Eq (21): L_DPLR = p1·L_rank-k + p2·L_diag
# ============================================================
def get_transformer_blocks(model: nn.Module) -> dict:
    """Return {name: block} of transformer blocks — works for both timm Swin
    (SwinTransformerBlock) and plain ViT/DeiT (vision_transformer.Block). Used by
    the DPLR per-block distillation loss, so it must span every supported family."""
    import timm.models.swin_transformer as _swin
    import timm.models.vision_transformer as _vit
    block_types = (_swin.SwinTransformerBlock, _vit.Block)
    return {n: m for n, m in model.named_modules() if isinstance(m, block_types)}


# Backward-compat alias: this used to be Swin-only and named get_swin_blocks.
get_swin_blocks = get_transformer_blocks


class DPLRBlockLoss(nn.Module):
    """
    Per-block DPLR-FIM loss for QAT.

    Initialization (once, after APB applied):
      For each transformer block, collect k pairs of (Δz, ∇L) by running k batches of
      forward+backward (CE loss) on the current APB model with FP teacher.
      Store: G shape (k, N), D shape (k, N), inv_B shape (k, k), diag shape (N,).

    During training (each batch):
      Run FP teacher forward (no grad) to get z_fp_b per block.
      Run APB model forward (with grad) to get z_apb_b per block (via hooks).
      For each block b:
        Δz_b = z_apb_b - z_fp_b                   shape (B, N)
        L_diag(b)   = E_B[ Σ_n diag_b[n] · Δz_b[B,n]² ]
        L_rank-k(b) = E_B[ (Δz_b · G_b^T) · inv_B_b · (G_b · Δz_b^T) ]
        L_DPLR(b)   = p1·L_rank-k(b) + p2·L_diag(b)
      Total = Σ_b L_DPLR(b)
    """
    def __init__(self, model_fp: nn.Module, k: int = 5,
                 p1: float = 1.0, p2: float = 1.0):
        super().__init__()
        self.model_fp = model_fp
        for p in self.model_fp.parameters():
            p.requires_grad = False
        self.model_fp.eval()
        self.k = k; self.p1 = p1; self.p2 = p2

        # Filled by initialize()
        self.states: dict = {}            # name → {'G','D','inv_B','diag','N'}
        # Caches filled by hooks each forward
        self.z_apb_cache: dict = {}
        self.z_fp_cache: dict  = {}
        self._installed = False

    def _make_cache_hook(self, cache, name):
        def hook(_m, _inp, out):
            cache[name] = out
        return hook

    def install_hooks(self, model_apb: nn.Module):
        if self._installed: return
        blocks_apb = get_transformer_blocks(model_apb)
        blocks_fp  = get_transformer_blocks(self.model_fp)
        assert set(blocks_apb) == set(blocks_fp), 'block names mismatch FP vs APB'
        for n, m in blocks_apb.items():
            m.register_forward_hook(self._make_cache_hook(self.z_apb_cache, n))
        for n, m in blocks_fp.items():
            m.register_forward_hook(self._make_cache_hook(self.z_fp_cache, n))
        self._installed = True

    def clear_caches(self):
        self.z_apb_cache.clear()
        self.z_fp_cache.clear()

    def initialize(self, model_apb: nn.Module, calib_loader, device):
        """Collect k FIM samples per block (paper-style, Section 3.3)."""
        self.install_hooks(model_apb)
        blocks_apb = get_transformer_blocks(model_apb)

        # Backward hooks to capture gradient at block output
        grad_cache = {}
        bwd_hooks = []
        for name, m in blocks_apb.items():
            def make_bwd(n):
                def h(_m, _gi, go):
                    if go[0] is not None:
                        grad_cache[n] = go[0].detach()
                return h
            bwd_hooks.append(m.register_full_backward_hook(make_bwd(name)))

        G_lists = {n: [] for n in blocks_apb}
        D_lists = {n: [] for n in blocks_apb}

        crit = nn.CrossEntropyLoss()
        it = iter(calib_loader)
        for _ in range(self.k):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(calib_loader); x, y = next(it)
            x = x.to(device); y = y.to(device)

            self.clear_caches(); grad_cache.clear()
            with torch.no_grad():
                _ = self.model_fp(x)
            model_apb.zero_grad()
            logits = model_apb(x)
            loss = crit(logits, y)
            loss.backward()

            for name in blocks_apb:
                if name not in grad_cache: continue
                z_a = self.z_apb_cache[name].detach()
                z_f = self.z_fp_cache[name].detach()
                g = grad_cache[name]
                B = z_a.shape[0]
                d = (z_a.reshape(B, -1) - z_f.reshape(B, -1)).abs().float().mean(dim=0)
                gv = g.reshape(B, -1).abs().float().mean(dim=0)
                G_lists[name].append(gv.cpu()); D_lists[name].append(d.cpu())

        for h in bwd_hooks: h.remove()
        model_apb.zero_grad()
        self.clear_caches()

        for name in list(blocks_apb):
            if not G_lists[name]: continue
            G = torch.stack(G_lists[name], dim=0).to(device)          # (k, N)
            D = torch.stack(D_lists[name], dim=0).to(device)          # (k, N)
            DD = D @ D.T                                              # (k, k)
            inv_B = torch.linalg.inv(DD + 1e-6 * torch.eye(self.k, device=device))
            self.states[name] = {
                'G': G, 'D': D, 'inv_B': inv_B,
                'diag': G.mean(dim=0),    # (N,)
                'N': G.shape[1],
            }

    def compute(self) -> torch.Tensor:
        """Sum L_DPLR over all blocks for current batch (caches must be populated)."""
        device = next(self.model_fp.parameters()).device
        total = torch.tensor(0.0, device=device)
        if not self.states:
            return total
        for name, s in self.states.items():
            if name not in self.z_apb_cache or name not in self.z_fp_cache:
                continue
            z_a = self.z_apb_cache[name]
            z_f = self.z_fp_cache[name].detach()
            B = z_a.shape[0]
            cha = (z_a.reshape(B, -1) - z_f.reshape(B, -1)).float()    # (B, N)

            # L_diag (paper Eq 14, simplified)
            L_diag = (cha.pow(2) * s['diag'].unsqueeze(0)).mean()

            # L_rank-k (paper Eq 18)
            A = cha @ s['G'].T                                          # (B, k)
            L_rank = (A.unsqueeze(1) @ s['inv_B'] @ A.unsqueeze(-1)).squeeze(-1).squeeze(-1).mean()

            total = total + self.p1 * L_rank + self.p2 * L_diag
        return total


# ============================================================
# EVAL / TRAIN UTILS
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    c1 = c5 = n = 0
    total = max_batches if max_batches is not None else len(loader)
    pbar = tqdm(loader, total=total, desc='eval', leave=False)
    for i, (x, y) in enumerate(pbar):
        if max_batches is not None and i >= max_batches: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, p5 = logits.topk(5, dim=1)
        c1 += (p5[:, 0] == y).sum().item()
        c5 += (p5 == y.unsqueeze(1)).any(dim=1).sum().item()
        n += y.size(0)
        if n > 0:
            pbar.set_postfix(top1=f'{100*c1/n:.2f}')
    return 100 * c1 / n, 100 * c5 / n, n


def avg_apb_stats(model):
    layers = [m for m in model.modules() if isinstance(m, APBLinear)]
    if not layers:
        return 0, 0.0, 0.0
    rb = sum(m.binary_ratio for m in layers) / len(layers)
    bits = sum(m.storage_bits_per_weight() for m in layers) / len(layers)
    return len(layers), rb, bits


def pack_apb_state_dict(model, fp_dtype='fp32'):
    """Pack per APB paper Eq 10 + Fig 2 (right):
        For each APB layer:
          binary_signs_packed: (n-s) × 1 bit, packed to uint8.
              Signs of weights at *non-FP positions only*, in canonical row-major order
              (skipping positions listed in fp_positions). Sign bit 1 → +α, 0 → -α.
          fp_positions: int32 array (s entries) — flat row-major indices of FP weights.
          fp_values:    fp_dtype array (s entries) — FP values at those positions.
          alpha:        scalar.

        Non-APB params (LN, biases of non-APB layers, head if scope=skip) saved as fp_dtype.

    Total bits = (n-s) + s·(b_v + b_p) + 32  per layer  (paper Eq 10).
    Position list (paper's choice) is more compact than bitmap mask when p < ~0.96.
    """
    fp_torch_dtype = torch.float16 if fp_dtype == 'fp16' else torch.float32
    packed = {'_apb_layers': {}, '_other': {}, '_meta': {'fp_dtype': fp_dtype}}

    for name, m in model.named_modules():
        if isinstance(m, APBLinear):
            w_flat = m.latent_weight.detach().cpu().view(-1)
            mask_flat = m.mask.detach().cpu().view(-1)
            shape = (m.out_features, m.in_features)

            fp_positions = torch.nonzero(~mask_flat, as_tuple=False).view(-1).to(torch.int32)
            fp_values = w_flat[~mask_flat].to(fp_torch_dtype)

            binary_signs = (w_flat[mask_flat] > 0).to(torch.uint8).numpy()
            binary_signs_packed = np.packbits(binary_signs)

            packed['_apb_layers'][name] = {
                'shape': shape,
                'binary_signs_packed': torch.from_numpy(binary_signs_packed),
                'n_binary': int(mask_flat.sum().item()),  # to truncate packbits padding on unpack
                'fp_positions': fp_positions,
                'fp_values': fp_values,
                'alpha': float(m.alpha.detach().abs().item()),
                'bias': m.bias.detach().to(fp_torch_dtype).cpu() if m.bias is not None else None,
            }

    apb_param_names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, APBLinear):
            apb_param_names.add(f'{name}.latent_weight')
            apb_param_names.add(f'{name}.alpha')
            apb_param_names.add(f'{name}.mask')
            if mod.bias is not None:
                apb_param_names.add(f'{name}.bias')

    for name, tensor in model.state_dict().items():
        if name in apb_param_names:
            continue
        packed['_other'][name] = tensor.detach().to(fp_torch_dtype).cpu()

    return packed


def packed_theoretical_bits(model, b_v: int = 32) -> dict:
    """Total bits per paper Eq 10, summed across APB layers only.

    Returns dict with per-layer + aggregate eff_bits/MB so the report
    matches paper Table II-III exactly (b_v=32 by default).
    """
    total_n = 0
    total_bits = 0
    per_layer = {}
    for name, m in model.named_modules():
        if isinstance(m, APBLinear):
            n = m.in_features * m.out_features
            b_p = math.ceil(math.log2(max(n - 1, 2))) + 1
            s = int((~m.mask).sum().item())
            layer_bits = (n - s) + s * (b_v + b_p) + 32  # +32 for alpha scalar
            per_layer[name] = {
                'n': n, 's': s, 'b_p': b_p, 'bits': layer_bits,
                'bits_per_weight': layer_bits / n,
            }
            total_n += n
            total_bits += layer_bits
    return {
        'per_layer': per_layer,
        'total_n_weights': total_n,
        'total_bits': total_bits,
        'avg_bits_per_weight': total_bits / max(total_n, 1),
        'apb_layers_MB': total_bits / 8 / 1e6,
    }


def actual_effective_bits(packed: dict, file_bytes: int) -> dict:
    """ACTUAL bits/weight measured from the packed dict + on-disk file size.

    Unlike packed_theoretical_bits (paper Eq 10, b_p-bit positions, APB-only),
    this reflects what was REALLY written:
      - apb_zone_bpw : bytes of the _apb_layers tensors only ÷ APB weights.
                       Higher than the 'theoretical' number because fp_positions
                       are stored as int32 (32 bit) instead of b_p≈20 bit.
      - whole_bpw    : full file size × 8 ÷ ALL model weights (APB + _other).
                       This is the honest "how many bits/weight did we ship"
                       (APB scope is always full: head.fc + downsample.reduction are
                       quantized too, only LN/biases remain fp32 in _other).
    """
    def _nbytes(t):
        return t.numel() * t.element_size() if torch.is_tensor(t) else 0

    apb_w = sum(L['shape'][0] * L['shape'][1] for L in packed['_apb_layers'].values())
    apb_bytes = sum(_nbytes(v) for L in packed['_apb_layers'].values()
                    for v in L.values())
    other_w = sum(t.numel() for t in packed['_other'].values() if torch.is_tensor(t))
    total_w = apb_w + other_w
    return {
        'apb_weights': apb_w,
        'other_weights': other_w,
        'total_weights': total_w,
        'apb_zone_bpw': apb_bytes * 8 / max(apb_w, 1),
        'whole_bpw': file_bytes * 8 / max(total_w, 1),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=str, default='swin_small_patch4_window7_224',
                   help='timm model name (default swin_small_patch4_window7_224). Also '
                        'supports plain ViT/DeiT, e.g. deit_small_patch16_224, '
                        'vit_small_patch16_224 — APB wrap, FIM, DPLR per-block loss and '
                        'pruning all generalise across the Swin/ViT families. Ignored '
                        'when --init-model loads a whole saved model object.')
    p.add_argument('--baseline-ckpt', type=str, default='',
                   help='Path to a finetuned FP baseline state_dict for THIS model '
                        '(overrides the default Swin ckpt/best_swin.pth). Use a per-arch '
                        'baseline, e.g. ckpt_deit/best.pth, so QAT starts from a trained '
                        'model. Empty = use default ckpt/best_swin.pth (Swin-S).')
    p.add_argument('--binary-ratio', type=float, default=0.75,
                   help='Fraction of weights to binarize (default 0.75)')
    p.add_argument('--partition', choices=['fisher', 'magnitude'], default='magnitude',
                   help='APB binary/FP partition criterion (which weights stay FP outliers). '
                        '"magnitude" = keep largest-|w| FP (standard APB; DEFAULT). '
                        '"fisher" = keep highest empirical-Fisher weights FP (FIM-guided, our method; '
                        'needs a FIM pass). Renamed from "fim" 2026-07-13: the ambiguous "fim" name '
                        '(defaulted to self-devised dplr) was dropped — only rigorous fisher.')
    p.add_argument('--act-bits', type=int, default=0,
                   help='Activation quant bit-width on the INPUT activations of each APB '
                        'Linear. 0 or 32 = OFF (weight-only APB, default, backward-compatible). '
                        '2..31 = LSQ (signed symmetric, learnable step); verify at 8 first, then '
                        'sweep 4/2. 1 = BinaryActQuant (sign·scale, XNOR/Bi-Real STE) — true 1-bit '
                        'activation; expect a large drop on ViT (two-sided, outlier-heavy acts).')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--fim-batches', type=int, default=5,
                   help='Number of calib batches (= rank k) for FIM extraction (default 5). '
                        'Also serves as rank k for the DPLR distillation loss.')
    p.add_argument('--importance-full', action='store_true',
                   help='Compute the APB-partition importance over the ENTIRE dataset '
                        '(one full pass) instead of --fim-batches calibration batches. '
                        'For exact importance with no sampling error (recommended with '
                        "--importance fisher). Independent of --fim-batches, which still "
                        'sets rank k for the DPLR loss.')
    p.add_argument('--importance', '--fim-mode', dest='importance', type=str,
                   default='fisher', choices=['fisher'],
                   help="FIM variant for partition='fisher'. Only 'fisher' (exact empirical Fisher) — "
                        "the self-devised 'dplr' variant was removed 2026-07-13. Ignored when "
                        "partition=magnitude. (The DPLR distillation loss builds its own FIM, independent.)")
    p.add_argument('--fim-p1', type=float, default=1.0,
                   help='Weight for rank-k (L2) component in DPLR (default 1.0)')
    p.add_argument('--fim-p2', type=float, default=1.0,
                   help='Weight for diag (L1) component in DPLR (default 1.0)')

    # Dynamic mask refresh
    p.add_argument('--recompute-fim-every', type=int, default=0,
                   help='Recompute DPLR-FIM + refresh mask every N epochs (0=disabled, '
                        'default frozen mask). Adds ~5-10s per refresh. '
                        'Keeps ratio fixed but partition adapts to updated weights.')

    # DPLR distillation loss
    p.add_argument('--use-dplr-loss', action='store_true',
                   help='Add per-block DPLR-FIM loss (knowledge distillation from FP teacher) '
                        'during QAT. Doubles memory (FP model in RAM) but tighter accuracy.')
    p.add_argument('--dplr-lambda', type=float, default=1.0,
                   help='Weight for DPLR loss vs CE loss (default 1.0). '
                        'total = CE + lambda · Σ_blocks L_DPLR')
    p.add_argument('--dataset', choices=['tiny', 'imagenet1k', 'cifar10', 'cifar100'],
       default='tiny',
                   help='"tiny" (default) = HF tiny-imagenet, 91k/9.1k after IN-1k remap, '
                        '~22 min/epoch on T4. '
                        '"imagenet1k" = full ImageNet-1k from --data-dir (ImageFolder layout), '
                        '~5h/epoch on T4 → only viable for PTQ or few-epoch QAT. '
                        '"cifar10"/"cifar100" = torchvision (auto-download, no HF datasets), '
                        '32px upsampled to 224; head re-created for 10/100 classes so the '
                        'FP baseline starts near-random and must be trained.')
    p.add_argument('--data-dir', type=str, default='',
                   help='Root dir for ImageNet-1k (must contain train/ and val/ ImageFolder layout). '
                        'Required when --dataset=imagenet1k. Ignored for --dataset=tiny.')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                   help='CUDA mixed-precision (fp16) training. On by default '
                        '(1.5-2x speedup on T4/RTX). Disable with --no-amp.')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--debug', action='store_true',
                   help='Quick mode: 1 epoch, 1 FIM batch, tiny val eval')
    p.add_argument('--out-dir', type=str, default=str(PROJECT_ROOT / 'checkpoints' / 'qat'))
    p.add_argument('--log-file', type=str, default='',
                   help='Path to log file. Default: <out_dir>/train.log. '
                        'Opened in append mode so resumed runs continue the same log. '
                        'Pass "none" to disable file logging.')
    p.add_argument('--resume', type=str, default='',
                   help='Path to last.pth checkpoint to resume from (model+opt+sched+scaler+epoch). '
                        'Use to recover after Kaggle session timeout. Mask is re-derived from '
                        'the saved checkpoint, so --binary-ratio must match the original run.')
    p.add_argument('--init-model', type=str, default='',
                   help='Path to a WHOLE saved model object (torch.save(model), NOT a '
                        'state_dict) to quantize as-is — e.g. a pruned model from '
                        'prune_cifar.py (ckpt/pruned_cifar100/best_pruned_model.pt). '
                        'Its (possibly pruned) architecture is used directly; '
                        'timm.create_model + init_from_baseline are SKIPPED. The model '
                        "must already match --dataset's num_classes. This is how you "
                        'quantize a pruned model (prune -> finetune -> APB-QAT).')
    p.add_argument('--export-packed', action=argparse.BooleanOptionalAction, default=True,
                   help='After training, also save best_packed.pt in paper format (Eq 10): '
                        '(n-s) binary signs + (fp_position, fp_value) pairs + α per layer. '
                        'Use --no-export-packed to disable.')
    p.add_argument('--packed-fp-dtype', choices=['fp32', 'fp16'], default='fp32',
                   help='Bit-width for FP zone values in packed file. fp32 matches paper Eq 10 '
                        '(b_v=32) exactly; fp16 halves the file size.')
    args = p.parse_args()

    if args.debug:
        args.epochs = 2; args.fim_batches = 1; args.batch_size = 32; args.num_workers = 0
        args.importance_full = False   # keep debug fast (no full-dataset importance pass)
        # 2 epochs so recompute-fim-every=1 can be tested if user passes it

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Windows consoles default to cp1252, which can't encode the 'α' (α) in our
    # log lines — force UTF-8 so real runs don't crash mid-training on a print().
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass

    # ----- Log file (tee stdout) -----
    log_path = args.log_file or str(out_dir / 'train.log')
    if log_path.lower() != 'none':
        log_fp = open(log_path, 'a', buffering=1, encoding='utf-8')
        sys.stdout = _Tee(sys.stdout, log_fp)
        print(f'\n[{datetime.now():%Y-%m-%d %H:%M:%S}] ===== run start (log → {log_path}) =====')

    print(f'='*60)
    print(f'APB QAT | model={args.model if not args.init_model else "(from --init-model)"} | device={device}')
    print(f'Args: {vars(args)}')
    print(f'='*60)

    # ----- Data (loaders imported lazily so CIFAR doesn't pull HF `datasets`) -----
    if args.dataset == 'tiny':
        from tiny_imagenet_loader import TinyImageNetLoaderGenerator
        g = TinyImageNetLoaderGenerator(val_batch_size=args.batch_size,
                                         num_workers=args.num_workers)
        num_classes = 1000  # tiny labels remapped onto IN-1k head
    elif args.dataset == 'imagenet1k':
        if not args.data_dir:
            raise ValueError('--data-dir is required when --dataset=imagenet1k')
        from imagenet1k_loader import ImageNet1kLoaderGenerator
        g = ImageNet1kLoaderGenerator(data_dir=args.data_dir,
                                       val_batch_size=args.batch_size,
                                       num_workers=args.num_workers)
        num_classes = 1000
    elif args.dataset in ('cifar10', 'cifar100'):
        from cifar_loader import CifarLoaderGenerator
        g = CifarLoaderGenerator(which=args.dataset, data_dir=args.data_dir,
                                 val_batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        num_classes = g.num_classes
    else:
        raise ValueError(f'unknown dataset {args.dataset!r}')
    train_loader = torch.utils.data.DataLoader(
        g.train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = g.val_loader()
    calib_loader = torch.utils.data.DataLoader(
        g.train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)

    # ----- Model -----
    if args.init_model:
        # Quantize a pre-built (e.g. pruned) model saved as a WHOLE object. Its
        # architecture may differ from the stock model (fewer heads / MLP neurons),
        # so we load it directly instead of create_model + load_state_dict.
        print(f'Loading whole model object from {args.init_model} ...')
        model = torch.load(args.init_model, map_location=device, weights_only=False)
        if not isinstance(model, nn.Module):
            raise TypeError(f'--init-model must be a saved nn.Module (torch.save(model)), '
                            f'got {type(model)}. For a state_dict, use ckpt/best_swin.pth via '
                            f'init_from_baseline instead.')
        head = getattr(model, 'head', None)
        n_out = getattr(getattr(head, 'fc', head), 'out_features', None)
        if n_out is not None and n_out != num_classes:
            raise ValueError(f'--init-model has {n_out} output classes but --dataset '
                             f'{args.dataset} needs {num_classes}. Mismatch.')
        print(f'>>> Quantizing pre-built model as-is (skipped create_model / '
              f'init_from_baseline){f", head out={n_out}" if n_out else ""}.')
    else:
        print(f'Loading {args.model} pretrained (num_classes={num_classes}) ...')
        model = timm.create_model(args.model, pretrained=True,
                                  num_classes=num_classes)
        baseline_path = args.baseline_ckpt or None
        status = init_from_baseline(model, baseline_path)
        shown = baseline_path or DEFAULT_INIT_CKPT
        if status == 'loaded':
            print(f'>>> Initialised from finetuned FP baseline: {shown}')
        elif status == 'none':
            print(f'>>> No {shown} — using timm-pretrained weights.')
    model.to(device).eval()

    # FP baseline
    max_b = 10 if args.debug else None
    t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
    print(f'[FP baseline]   top1={t1:.2f}% top5={t5:.2f}% ({n} samples)')

    # ----- FIM extraction (needed for partition='fisher' OR the DPLR distillation loss) -----
    # DPLR loss builds its own per-block FIM inside DPLRBlockLoss.initialize() and does
    # NOT consume this weight-FIM dict — so only partition='fisher' actually needs it.
    # (Previously `or args.use_dplr_loss` triggered a FIM pass that was computed then
    # discarded — wasteful, and a FULL pass with --importance-full + magnitude.)
    need_fim = (args.partition == 'fisher')
    fim_dict = {}
    if need_fim:
        over = ('the FULL dataset' if args.importance_full
                else f'{args.fim_batches} calib batches')
        print(f'Computing {args.importance.upper()}-FIM(W) over {over} '
              f"(p1={args.fim_p1}, p2={args.fim_p2}) [for partition='fisher'] ...")
        t0 = time.time()
        fim_dict = compute_weight_dplr_fim(model, calib_loader, device,
                                            n_batches=args.fim_batches,
                                            p1=args.fim_p1, p2=args.fim_p2,
                                            fim_mode=args.importance,
                                            scope='full',
                                            full=args.importance_full)
        print(f'FIM done in {time.time()-t0:.1f}s, {len(fim_dict)} layers (scope=full)')
    else:
        print(f"Skipping FIM extraction (partition={args.partition!r}).")

    # Snapshot the FP model BEFORE APB, so a pruned --init-model can serve as its
    # own distillation teacher (deepcopy of the pruned FP net, pre-quantization).
    fp_teacher_snapshot = (copy.deepcopy(model)
                           if (args.use_dplr_loss and args.init_model) else None)

    # ----- Apply APB -----
    print(f'Applying APB (scope=full, binary_ratio={args.binary_ratio}, '
          f'partition={args.partition}, act_bits={args.act_bits or "off"}) ...')
    model = apply_apb(model, fim_dict, args.binary_ratio, device, scope='full',
                      act_bits=args.act_bits, partition=args.partition)
    nl, rb, bits = avg_apb_stats(model)
    print(f'APB: {nl} layers wrapped | avg binary={rb:.3f} | avg eff_bits={bits:.2f}')
    n_actq = sum(1 for m in model.modules()
                 if isinstance(m, (LSQActQuant, BinaryActQuant)))
    if n_actq:
        kind = 'Binary (sign·scale)' if args.act_bits == 1 else 'LSQ (signed)'
        print(f'Activation quant: {kind} {args.act_bits}-bit on {n_actq} '
              f'APB-Linear inputs')

    # Eval right after APB (before training)
    t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
    print(f'[Post-APB]      top1={t1:.2f}% top5={t5:.2f}% ({n} samples) — before QAT')

    # ----- (optional) DPLR-FIM distillation loss setup -----
    dplr = None
    if args.use_dplr_loss:
        print('Setting up DPLR-FIM per-block loss (FP teacher + FIM init) ...')
        if fp_teacher_snapshot is not None:
            # Pruned student: teacher = the pruned FP net itself (pre-APB snapshot).
            model_fp = fp_teacher_snapshot
        else:
            model_fp = timm.create_model(args.model, pretrained=True,
                                         num_classes=num_classes)
            init_from_baseline(model_fp, args.baseline_ckpt or None)  # teacher matches student's init
        model_fp.to(device).eval()
        dplr = DPLRBlockLoss(model_fp, k=args.fim_batches,
                             p1=args.fim_p1, p2=args.fim_p2)
        dplr.initialize(model, calib_loader, device)
        print(f'DPLR ready: {len(dplr.states)} transformer blocks tracked, '
              f'λ={args.dplr_lambda}, p1={args.fim_p1}, p2={args.fim_p2}')

    # ----- QAT -----
    # Paper APB convention: weight_decay applies to weights/bias only, NOT to α.
    # Excluding α prevents WD from pulling it toward 0 (which would kill binary magnitude).
    # No weight-decay on α (binary magnitude) nor on the LSQ activation step.
    def _is_nowd(n):
        return ('alpha' in n or 'act_quant.step' in n
                or 'act_quant.scale' in n)
    nowd_params  = [p for n, p in model.named_parameters() if _is_nowd(n)]
    other_params = [p for n, p in model.named_parameters() if not _is_nowd(n)]
    opt = optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': nowd_params, 'weight_decay': 0.0},
    ], lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    best_top1 = 0.0
    freeze_at = max(args.epochs // 2, 1)

    use_amp = bool(args.amp) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        if use_amp and 'scaler' in ck:
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        best_top1 = ck.get('best_top1', 0.0)
        print(f'>>> Resumed from {args.resume}: epoch {ck["epoch"]+1}/{args.epochs}, '
              f'best_top1={best_top1:.2f}%')
        if start_epoch >= args.epochs:
            print(f'>>> Already finished ({start_epoch} >= {args.epochs}). Nothing to do.')
            return

    print(f'='*60)
    print(f'QAT training: {args.epochs} epochs (start={start_epoch}), lr={args.lr}, '
          f'freeze α at epoch {freeze_at}, dplr={dplr is not None}, amp={use_amp}')
    print(f'='*60)
    for ep in range(start_epoch, args.epochs):
        # Periodic mask refresh (FIM recompute only when partition='fisher')
        if (args.recompute_fim_every > 0 and ep > 0
                and ep % args.recompute_fim_every == 0):
            t_fim = time.time()
            fim_new = {}
            if args.partition == 'fisher':
                fim_new = compute_weight_dplr_fim(model, calib_loader, device,
                                                  n_batches=args.fim_batches,
                                                  p1=args.fim_p1, p2=args.fim_p2,
                                                  fim_mode=args.importance,
                                                  scope='full',
                                                  full=args.importance_full)
            n_upd, flip_pct = update_masks_from_fim(model, fim_new, args.binary_ratio,
                                                     scope='full',
                                                     partition=args.partition)
            print(f'  >> Epoch {ep+1}: refreshed mask [{args.partition}] ({n_upd} layers, '
                  f'{flip_pct:.1f}% positions flipped) in {time.time()-t_fim:.1f}s')

        if ep == freeze_at:
            for m in model.modules():
                if isinstance(m, APBLinear):
                    m.alpha.requires_grad = False
            print(f'  >> Epoch {ep+1}: α frozen (latent_weight still trainable)')

        model.train()
        t0 = time.time(); loss_sum = 0; ce_sum = 0; dplr_sum = 0; loss_n = 0
        max_train_batches = 6 if args.debug else len(train_loader)
        pbar = tqdm(train_loader, total=max_train_batches,
                    desc=f'ep {ep+1}/{args.epochs}', leave=True)
        for i, (x, y) in enumerate(pbar):
            opt.zero_grad()
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            if dplr is not None:
                dplr.clear_caches()
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                    _ = dplr.model_fp(x)        # populates z_fp_cache

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)                # populates z_apb_cache (if hooks installed)
                loss_ce = crit(logits, y)

                loss_dplr_val = 0.0
                if dplr is not None:
                    loss_dplr = dplr.compute()
                    loss = loss_ce + args.dplr_lambda * loss_dplr
                    loss_dplr_val = loss_dplr.detach().item()
                    dplr_sum += loss_dplr_val * y.size(0)
                else:
                    loss = loss_ce

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * y.size(0)
            ce_sum   += loss_ce.item() * y.size(0)
            loss_n   += y.size(0)
            if dplr is not None:
                pbar.set_postfix(loss=f'{loss.item():.3f}',
                                 ce=f'{loss_ce.item():.3f}',
                                 dplr=f'{args.dplr_lambda * loss_dplr_val:.4f}')
            else:
                pbar.set_postfix(loss=f'{loss.item():.3f}')
            if args.debug and i >= 5: break
        sched.step()
        train_loss = loss_sum / loss_n
        train_ce   = ce_sum / loss_n
        train_dplr = dplr_sum / loss_n if dplr is not None else 0.0

        t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
        dt = time.time() - t0
        if dplr is not None:
            print(f'Ep {ep+1}/{args.epochs}: loss={train_loss:.4f} '
                  f'(ce={train_ce:.4f} + λ·dplr={args.dplr_lambda * train_dplr:.4f}) | '
                  f'val top1={t1:.2f}% top5={t5:.2f}% | {dt:.1f}s')
        else:
            print(f'Ep {ep+1}/{args.epochs}: train_loss={train_loss:.4f} | '
                  f'val top1={t1:.2f}% top5={t5:.2f}% | {dt:.1f}s')

        if t1 > best_top1:
            best_top1 = t1
            torch.save(model.state_dict(), out_dir / 'best.pth')

        # Save resumable checkpoint every epoch
        torch.save({
            'epoch': ep,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'scaler': scaler.state_dict() if use_amp else None,
            'best_top1': best_top1,
        }, out_dir / 'last.pth')

    print(f'='*60)
    print(f'DONE. Best val top1 = {best_top1:.2f}%')
    print(f'='*60)

    # ----- Optional packed export (paper Eq 10 / Fig 2 format) -----
    if args.export_packed:
        best_pth = out_dir / 'best.pth'
        if best_pth.exists():
            model.load_state_dict(torch.load(best_pth, map_location=device))
        b_v_paper = 32 if args.packed_fp_dtype == 'fp32' else 16
        report = packed_theoretical_bits(model, b_v=b_v_paper)
        packed = pack_apb_state_dict(model, fp_dtype=args.packed_fp_dtype)
        packed_path = out_dir / 'best_packed.pt'
        torch.save(packed, packed_path)
        fp32_bytes = best_pth.stat().st_size if best_pth.exists() else 0
        packed_bytes = packed_path.stat().st_size
        act = actual_effective_bits(packed, packed_bytes)
        print(f'='*60)
        print(f'Packed export (paper Eq 10 format, b_v={b_v_paper}): {packed_path}')
        print(f'  >> Effective bit (WHOLE model): {act["whole_bpw"]:.2f} bits/weight '
              f'over {act["total_weights"]/1e6:.1f}M params')
        print(f'  Actual file: best_packed.pt = {packed_bytes/1e6:.1f} MB '
              f'(incl. non-APB params fp32: LN, biases, head/downsample if scope=skip)')
        print(f'  best.pth (training FP32): {fp32_bytes/1e6:.1f} MB '
              f'| Compression: {fp32_bytes/max(packed_bytes,1):.1f}×')
        print(f'  (paper-compare only — Eq10 theoretical, APB layers: '
              f'{report["avg_bits_per_weight"]:.2f} bits/weight)')
        print(f'='*60)


if __name__ == '__main__':
    main()
