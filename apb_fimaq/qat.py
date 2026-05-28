"""
APB QAT on Swin-S — DPLR-FIM-based partition + end-to-end fine-tuning.

Importance ranking uses FIMA-Q's DPLR-FIM (Diagonal + Low-Rank, paper Eq 20-21):
    F_DPLR(W_ij) = (p1 · E[(∇z_i)²]  +  p2 · E[|∇z_i|]) · E[x_j²]
                    ↑ rank-k (L2)       ↑ diag (L1)      ↑ activation
    Mask = (F_DPLR < τ percentile)

Flow:
  1. Load pretrained Swin-S
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
"""
import argparse
import sys
from tqdm.auto import tqdm
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from tiny_imagenet_loader import TinyImageNetLoaderGenerator  # noqa: E402


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


class APBLinear(nn.Module):
    """nn.Linear wrapper with APB partition.

    Forward (per weight w_ij):
        mask[i,j] = True   →  α · sign(w_ij)       (1-bit binary)
        mask[i,j] = False  →  w_ij                  (FP32, learnable)
    Backward: STE — both α and latent_weight receive correct gradients.
    """
    def __init__(self, linear: nn.Linear, mask: torch.Tensor):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.latent_weight = nn.Parameter(linear.weight.data.clone())
        self.bias = linear.bias
        # α: scalar learnable, init from mean|w|; kept positive via abs() in forward
        self.alpha = nn.Parameter(self.latent_weight.detach().abs().mean())
        # Mask: True = binary. Frozen throughout training.
        self.register_buffer('mask', mask.bool())

    def forward(self, x):
        # alpha.abs() to keep binary magnitude positive (paper convention).
        # Grad flows correctly through abs (sign(alpha) backward).
        eff_w = _APBSTE.apply(self.latent_weight, self.alpha.abs(), self.mask)
        return F.linear(x, eff_w, self.bias)

    @property
    def binary_ratio(self) -> float:
        return self.mask.float().mean().item()

    def storage_bits_per_weight(self) -> float:
        p = self.binary_ratio
        return 1.0 * p + 32.0 * (1.0 - p)


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
                             scope='skip'):
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
      n_batches: number of (forward + backward) samples
      p1: weight for rank-k (L2) component
      p2: weight for diag (L1) component
      fim_mode: 'dplr' | 'rank' | 'diag' — switch for ablation

    Returns:
      dict {name: F_DPLR(W) tensor of shape (out, in)}
    """
    targets = get_target_layers(model, scope=scope)
    # Accumulators
    x_sq  = {n: 0.0 for n in targets}   # E[x²] per input channel
    g_abs = {n: 0.0 for n in targets}   # E[|g|]  per output channel  (for diag)
    g_sq  = {n: 0.0 for n in targets}   # E[g²]   per output channel  (for rank-k)
    n_samples = 0

    hooks = []
    for name, mod in targets.items():
        def make_fwd(n):
            def h(_m, inp, _out):
                x = inp[0].detach()
                B = x.shape[0]
                x_sq[n] = x_sq[n] + (x.float() ** 2).reshape(-1, x.shape[-1]).mean(dim=0) * B
            return h
        def make_bwd(n):
            def h(_m, _gi, go):
                g = go[0]
                if g is None: return
                B = g.shape[0]
                gf = g.detach().float()
                gf_flat = gf.reshape(-1, gf.shape[-1])
                g_abs[n] = g_abs[n] + gf_flat.abs().mean(dim=0) * B
                g_sq[n]  = g_sq[n]  + (gf_flat ** 2).mean(dim=0) * B
            return h
        hooks.append(mod.register_forward_hook(make_fwd(name)))
        hooks.append(mod.register_full_backward_hook(make_bwd(name)))

    model.train()
    crit = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(tqdm(calib_loader, total=n_batches, desc='FIM extract', leave=False)):
        if i >= n_batches: break
        model.zero_grad()
        x = x.to(device); y = y.to(device)
        loss = crit(model(x), y)
        loss.backward()
        n_samples += x.size(0)
    for h in hooks: h.remove()
    model.zero_grad()

    fim = {}
    for name in targets:
        if isinstance(g_abs[name], float):
            continue
        gd = g_abs[name] / n_samples            # f_diag (L1)
        gr = g_sq[name]  / n_samples            # f_rank-k (L2)
        xs = x_sq[name]  / n_samples            # E[x²]

        if   fim_mode == 'diag': f_i = gd
        elif fim_mode == 'rank': f_i = gr
        elif fim_mode == 'dplr': f_i = p1 * gr + p2 * gd
        else: raise ValueError(f'fim_mode must be dplr|rank|diag, got {fim_mode}')

        fim[name] = (f_i.unsqueeze(1) * xs.unsqueeze(0)).cpu()  # (out, in)
    return fim


# Backwards-compat alias
compute_weight_fim = compute_weight_dplr_fim


# ============================================================
# APPLY APB (replace nn.Linear → APBLinear)
# ============================================================
def apply_apb(model, fim_dict, binary_ratio, device, scope='skip'):
    """Wrap each target Linear with APBLinear using FIM-percentile mask."""
    targets = get_target_layers(model, scope=scope)
    for name, mod in targets.items():
        if isinstance(mod, APBLinear):
            continue  # already wrapped
        fim = fim_dict[name]
        tau = torch.quantile(fim.flatten().float(), binary_ratio)
        mask = (fim < tau)
        new_layer = APBLinear(mod, mask=mask).to(device)
        # Reassign in parent
        parent = model
        parts = name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_layer)
    return model


def update_masks_from_fim(model, fim_dict, binary_ratio, scope='skip'):
    """Refresh masks of APBLinear layers in-place using new FIM values.
    Keeps latent_weight, α, and structure intact — only mask buffer changes."""
    n_updated = 0
    n_flipped_total = 0
    n_weights_total = 0
    for name, mod in get_target_layers(model, scope=scope).items():
        if not isinstance(mod, APBLinear):
            continue
        if name not in fim_dict:
            continue
        fim = fim_dict[name].to(mod.mask.device)
        tau = torch.quantile(fim.flatten().float(), binary_ratio)
        new_mask = (fim < tau)
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
def get_swin_blocks(model: nn.Module) -> dict:
    """Return {name: SwinTransformerBlock} from a timm Swin model."""
    import timm.models.swin_transformer as _swin
    return {n: m for n, m in model.named_modules()
            if isinstance(m, _swin.SwinTransformerBlock)}


class DPLRBlockLoss(nn.Module):
    """
    Per-block DPLR-FIM loss for QAT.

    Initialization (once, after APB applied):
      For each Swin block, collect k pairs of (Δz, ∇L) by running k batches of
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
        blocks_apb = get_swin_blocks(model_apb)
        blocks_fp  = get_swin_blocks(self.model_fp)
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
        blocks_apb = get_swin_blocks(model_apb)

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


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--binary-ratio', type=float, default=0.75,
                   help='Fraction of weights to binarize (default 0.75)')
    p.add_argument('--apb-scope', choices=['skip', 'full'], default='skip',
                   help='Layer scope for APB. '
                        '"skip" = 96 Linears (default; safe, skips head + 3 downsample + patch_embed). '
                        '"full" = 100 Linears (aggressive; includes head.fc + downsample.reduction).')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--fim-batches', type=int, default=5,
                   help='Number of batches (= rank k) for FIM extraction (default 5)')
    p.add_argument('--fim-mode', type=str, default='dplr',
                   choices=['dplr', 'rank', 'diag'],
                   help='FIM approximation: dplr (paper default, p1·rank-k + p2·diag), '
                        'rank (only L2 sample variance), diag (only L1, faster)')
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
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--amp', action='store_true',
                   help='Enable CUDA mixed-precision (fp16) training. '
                        '1.5-2x speedup on T4/RTX, slight memory savings.')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--debug', action='store_true',
                   help='Quick mode: 1 epoch, 1 FIM batch, tiny val eval')
    p.add_argument('--out-dir', type=str, default=str(PROJECT_ROOT / 'checkpoints' / 'qat'))
    args = p.parse_args()

    if args.debug:
        args.epochs = 2; args.fim_batches = 1; args.batch_size = 32; args.num_workers = 0
        # 2 epochs so recompute-fim-every=1 can be tested if user passes it

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'='*60)
    print(f'APB QAT on Swin-S | device={device}')
    print(f'Args: {vars(args)}')
    print(f'='*60)

    # ----- Data -----
    g = TinyImageNetLoaderGenerator(val_batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = torch.utils.data.DataLoader(
        g.train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = g.val_loader()
    calib_loader = torch.utils.data.DataLoader(
        g.train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)

    # ----- Model -----
    print('Loading Swin-S pretrained ...')
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model.to(device).eval()

    # FP baseline
    max_b = 10 if args.debug else None
    t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
    print(f'[FP baseline]   top1={t1:.2f}% top5={t5:.2f}% ({n} samples)')

    # ----- FIM extraction (DPLR by default per FIMA-Q paper) -----
    print(f'Computing {args.fim_mode.upper()}-FIM(W) over {args.fim_batches} samples '
          f'(p1={args.fim_p1}, p2={args.fim_p2}) ...')
    t0 = time.time()
    fim_dict = compute_weight_dplr_fim(model, calib_loader, device,
                                        n_batches=args.fim_batches,
                                        p1=args.fim_p1, p2=args.fim_p2,
                                        fim_mode=args.fim_mode,
                                        scope=args.apb_scope)
    print(f'FIM done in {time.time()-t0:.1f}s, {len(fim_dict)} layers '
          f'(scope={args.apb_scope})')

    # ----- Apply APB -----
    print(f'Applying APB (scope={args.apb_scope}, binary_ratio={args.binary_ratio}) ...')
    model = apply_apb(model, fim_dict, args.binary_ratio, device, scope=args.apb_scope)
    nl, rb, bits = avg_apb_stats(model)
    print(f'APB: {nl} layers wrapped | avg binary={rb:.3f} | avg eff_bits={bits:.2f}')

    # Eval right after APB (before training)
    t1, t5, n = evaluate(model, val_loader, device, max_batches=max_b)
    print(f'[Post-APB]      top1={t1:.2f}% top5={t5:.2f}% ({n} samples) — before QAT')

    # ----- (optional) DPLR-FIM distillation loss setup -----
    dplr = None
    if args.use_dplr_loss:
        print('Setting up DPLR-FIM per-block loss (FP teacher + FIM init) ...')
        model_fp = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        model_fp.to(device).eval()
        dplr = DPLRBlockLoss(model_fp, k=args.fim_batches,
                             p1=args.fim_p1, p2=args.fim_p2)
        dplr.initialize(model, calib_loader, device)
        print(f'DPLR ready: {len(dplr.states)} Swin blocks tracked, '
              f'λ={args.dplr_lambda}, p1={args.fim_p1}, p2={args.fim_p2}')

    # ----- QAT -----
    # Paper APB convention: weight_decay applies to weights/bias only, NOT to α.
    # Excluding α prevents WD from pulling it toward 0 (which would kill binary magnitude).
    alpha_params  = [p for n, p in model.named_parameters() if 'alpha' in n]
    other_params  = [p for n, p in model.named_parameters() if 'alpha' not in n]
    opt = optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': alpha_params, 'weight_decay': 0.0},
    ], lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    best_top1 = 0.0
    freeze_at = max(args.epochs // 2, 1)

    use_amp = bool(args.amp) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f'='*60)
    print(f'QAT training: {args.epochs} epochs, lr={args.lr}, '
          f'freeze α at epoch {freeze_at}, dplr={dplr is not None}, amp={use_amp}')
    print(f'='*60)
    for ep in range(args.epochs):
        # Periodic FIM recompute + mask refresh
        if (args.recompute_fim_every > 0 and ep > 0
                and ep % args.recompute_fim_every == 0):
            t_fim = time.time()
            fim_new = compute_weight_dplr_fim(model, calib_loader, device,
                                              n_batches=args.fim_batches,
                                              p1=args.fim_p1, p2=args.fim_p2,
                                              fim_mode=args.fim_mode,
                                              scope=args.apb_scope)
            n_upd, flip_pct = update_masks_from_fim(model, fim_new, args.binary_ratio,
                                                     scope=args.apb_scope)
            print(f'  >> Epoch {ep+1}: refreshed FIM ({n_upd} layers, '
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
                                 dplr=f'{loss_dplr_val:.4f}')
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
                  f'(ce={train_ce:.4f} + λ·dplr={train_dplr:.4f}) | '
                  f'val top1={t1:.2f}% top5={t5:.2f}% | {dt:.1f}s')
        else:
            print(f'Ep {ep+1}/{args.epochs}: train_loss={train_loss:.4f} | '
                  f'val top1={t1:.2f}% top5={t5:.2f}% | {dt:.1f}s')

        if t1 > best_top1:
            best_top1 = t1
            torch.save(model.state_dict(), out_dir / 'best.pth')

    print(f'='*60)
    print(f'DONE. Best val top1 = {best_top1:.2f}%')
    print(f'='*60)


if __name__ == '__main__':
    main()
