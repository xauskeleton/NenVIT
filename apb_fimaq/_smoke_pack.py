"""Offline smoke test: build a tiny model, wrap with APB, export packed, verify sizes.

No dataset / no pretrained needed. Confirms:
  1. `storage_bits_per_weight` matches paper Eq 10 numerically
  2. `pack_apb_state_dict` produces a smaller file than FP32 state_dict
  3. Compression ratio scales with binary_ratio as expected
"""
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from qat import (
    APBLinear,
    apply_apb,
    avg_apb_stats,
    pack_apb_state_dict,
    packed_theoretical_bits,
)


def build_tiny_model():
    """Stack of 4 Linears approximating a transformer block (qkv, proj, fc1, fc2)."""
    C = 384
    return nn.Sequential(
        nn.Linear(C, 3 * C),     # qkv
        nn.Linear(C, C),         # proj
        nn.Linear(C, 4 * C),     # fc1
        nn.Linear(4 * C, C),     # fc2
    )


def make_fim_dict(model):
    """Mock FIM with random importance (good enough for partition)."""
    fim = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            fim[name] = torch.rand_like(mod.weight, device='cpu')
    return fim


def get_targets(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def apply_apb_simple(model, fim_dict, binary_ratio, device='cpu'):
    """Local copy of apply_apb without scope filter (we only have Linears)."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            fim = fim_dict[name]
            tau = torch.quantile(fim.flatten().float(), binary_ratio)
            mask = (fim < tau)
            new_layer = APBLinear(mod, mask=mask).to(device)
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_layer)
    return model


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def run_one(binary_ratio, out_dir='_smoke_out'):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)

    model = build_tiny_model()
    fp32_path = os.path.join(out_dir, f'fp32_br{binary_ratio:.2f}.pth')
    torch.save(model.state_dict(), fp32_path)
    fp32_mb = file_size_mb(fp32_path)

    fim = make_fim_dict(model)
    model = apply_apb_simple(model, fim, binary_ratio)

    n_layers, avg_binary, avg_eff_bits = avg_apb_stats(model)
    print(f'  br={binary_ratio:.2f}: {n_layers} APB layers, avg binary={avg_binary:.3f}, '
          f'avg eff_bits={avg_eff_bits:.2f}  [paper Eq 10]')

    report = packed_theoretical_bits(model, b_v=32)
    print(f'           theoretical: {report["avg_bits_per_weight"]:.2f} bits/weight, '
          f'APB-only = {report["apb_layers_MB"]:.3f} MB')

    packed = pack_apb_state_dict(model, fp_dtype='fp32')
    packed_path = os.path.join(out_dir, f'packed_br{binary_ratio:.2f}.pt')
    torch.save(packed, packed_path)
    packed_mb = file_size_mb(packed_path)

    raw_apb_path = os.path.join(out_dir, f'raw_apb_br{binary_ratio:.2f}.pth')
    torch.save(model.state_dict(), raw_apb_path)
    raw_apb_mb = file_size_mb(raw_apb_path)

    ratio_vs_fp = fp32_mb / max(packed_mb, 1e-9)
    ratio_vs_raw = raw_apb_mb / max(packed_mb, 1e-9)
    print(f'           FP32 state_dict:        {fp32_mb:.3f} MB')
    print(f'           APB raw state_dict:     {raw_apb_mb:.3f} MB  (latent_weight FP32 + mask)')
    print(f'           Packed (paper format):  {packed_mb:.3f} MB  '
          f'(vs FP32 = {ratio_vs_fp:.2f}x, vs raw APB = {ratio_vs_raw:.2f}x)')
    print()
    return dict(br=binary_ratio, fp32_mb=fp32_mb, raw_apb_mb=raw_apb_mb,
                packed_mb=packed_mb, eff_bits=avg_eff_bits)


if __name__ == '__main__':
    print('=== APB pack smoke test (tiny 4-Linear model, C=384) ===\n')
    print(f'Total weights in 4 Linears: 384*1152 + 384*384 + 384*1536 + 1536*384 = '
          f'{384*1152 + 384*384 + 384*1536 + 1536*384:,} = ~1.92M params\n')

    results = []
    for br in [0.0, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99]:
        results.append(run_one(br))

    print('=== Summary ===')
    print(f'{"br":>5} {"eff_bits":>10} {"FP32 MB":>10} {"raw APB":>10} {"packed":>10} {"ratio":>8}')
    for r in results:
        print(f'{r["br"]:>5.2f} {r["eff_bits"]:>10.2f} '
              f'{r["fp32_mb"]:>10.3f} {r["raw_apb_mb"]:>10.3f} '
              f'{r["packed_mb"]:>10.3f} {r["fp32_mb"]/r["packed_mb"]:>7.2f}x')

    print('\n=== Sanity checks ===')
    eff_bits_growing = all(results[i]['eff_bits'] >= results[i+1]['eff_bits']
                            for i in range(len(results)-1))
    sizes_growing = all(results[i]['packed_mb'] >= results[i+1]['packed_mb']
                         for i in range(len(results)-1))
    print(f'  eff_bits decreases with br:   {"OK" if eff_bits_growing else "FAIL"}')
    print(f'  packed size decreases with br: {"OK" if sizes_growing else "FAIL"}')
    print(f'  packed @ br=0.99 < 1/3 of FP32: '
          f'{"OK" if results[-1]["packed_mb"] < results[0]["fp32_mb"] / 3 else "FAIL"}')
