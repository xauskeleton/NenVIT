"""
Replace AdaRound weight quantizer with APBWeightQuantizer on selected Linears.

Targets = 96 Linears matching APB criteria:
    qkv, attn.proj, mlp.fc1, mlp.fc2  ×  24 Swin blocks
Skipped:
    patch_embed (Conv2d), downsample.reduction, head, all LayerNorms.

This module:
  - filters APB target modules from a wrapped FIMA-Q model
  - replaces each target's w_quantizer with APBWeightQuantizer
  - applies binary mask from FIM importance (per-layer percentile threshold)
"""
import logging
import torch
import torch.nn as nn

from apb_weight_quantizer import APBWeightQuantizer

# Match FIMA-Q quant_layers without modifying them
import sys
from pathlib import Path
_FIMA_ROOT = Path(__file__).resolve().parents[1] / 'FIMA-Q' / 'FIMA-Q'
if str(_FIMA_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIMA_ROOT))
from quant_layers.linear import MinMaxQuantLinear  # noqa: E402

APB_SKIP_PATTERNS = ('downsample.reduction', 'head', 'patch_embed')


def get_apb_target_modules(model: nn.Module) -> dict:
    """Return {name: quant_linear_module} for APB target Linears."""
    targets = {}
    for name, m in model.named_modules():
        if not isinstance(m, MinMaxQuantLinear):
            continue
        if any(p in name for p in APB_SKIP_PATTERNS):
            continue
        targets[name] = m
    return targets


def replace_weight_quantizers_with_apb(model: nn.Module,
                                        fim_dict: dict,
                                        binary_ratio: float = 0.75,
                                        per_role_ratio: dict = None) -> dict:
    """
    For each APB target Linear:
      1. Build APBWeightQuantizer matching the layer's n_V grouping (3 for qkv, 1 otherwise)
      2. Compute mask from FIM importance: mask[i,j] = True if F_diag(W_ij) < τ
         where τ = percentile(fim, binary_ratio_for_this_layer)
      3. Replace module.w_quantizer with the APB one

    Args:
      fim_dict: {name: F_diag(W) tensor}  — same shape as module.weight
      binary_ratio: default ratio (e.g. 0.75 = 75% binary)
      per_role_ratio: optional dict like {'qkv': 0.60, 'proj': 0.75, 'fc1': 0.85, 'fc2': 0.75}

    Returns: dict of stats per replaced layer.
    """
    per_role_ratio = per_role_ratio or {}
    targets = get_apb_target_modules(model)
    if not set(targets) <= set(fim_dict):
        missing = set(targets) - set(fim_dict)
        raise RuntimeError(f'Missing FIM for {len(missing)} layers, e.g.: {next(iter(missing))}')

    stats = {}
    for name, mod in targets.items():
        # Determine role and binary_ratio for this layer
        if '.attn.qkv' in name: role = 'qkv'
        elif '.attn.proj' in name: role = 'proj'
        elif '.mlp.fc1' in name: role = 'fc1'
        elif '.mlp.fc2' in name: role = 'fc2'
        else: role = 'other'
        ratio = per_role_ratio.get(role, binary_ratio)

        # Reshape weight to FIMA-Q's (n_V, crb_rows, in_features) layout
        n_V = getattr(mod, 'n_V', 1)
        crb_rows = getattr(mod, 'crb_rows', mod.out_features)
        w_reshaped = mod.weight.data.view(n_V, crb_rows, mod.in_features)

        # Build APB quantizer with same shape; copy α, δ stats from this view
        apb_q = APBWeightQuantizer(w_reshaped, n_V=n_V)

        # Reshape FIM to match weight view
        fim_W = fim_dict[name].to(mod.weight.device)
        fim_reshaped = fim_W.view(n_V, crb_rows, mod.in_features)

        # Build binary mask via per-layer percentile threshold
        apb_q.set_mask_from_fim(fim_reshaped, binary_ratio=ratio)
        apb_q.to(mod.weight.device)

        # Replace
        mod.w_quantizer = apb_q

        stats[name] = {
            'role': role,
            'n_V': n_V,
            'binary_ratio_requested': ratio,
            'binary_ratio_actual': apb_q.binary_ratio,
            'alpha': apb_q.apb_alpha.tolist(),
            'eff_bits': apb_q.storage_bits_per_weight(),
        }

    return stats


def log_apb_stats(stats: dict, top_n: int = 5):
    """Pretty-print summary of APB replacement."""
    by_role = {}
    for name, s in stats.items():
        by_role.setdefault(s['role'], []).append((name, s))

    logging.info(f'APB replaced w_quantizer in {len(stats)} layers:')
    for role, items in by_role.items():
        ratios = [s['binary_ratio_actual'] for _, s in items]
        eff = [s['eff_bits'] for _, s in items]
        logging.info(f'  [{role}] {len(items)} layers | '
                     f'binary_ratio mean={sum(ratios)/len(ratios):.3f} | '
                     f'eff_bits mean={sum(eff)/len(eff):.2f}')
    # Show sample
    sample = list(stats.items())[:top_n]
    for name, s in sample:
        logging.info(f'  e.g. {name}: ratio={s["binary_ratio_actual"]:.3f}, eff_bits={s["eff_bits"]:.2f}')
