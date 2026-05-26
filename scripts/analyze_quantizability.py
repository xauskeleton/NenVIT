"""
Per-layer quantization feasibility analysis for Swin-S.

For each layer, decides:
  - Can WEIGHT be quantized? (Y/N + suggested bits + risk)
  - Can ACTIVATION input be quantized? (Y/N + special handling)
  - Stats: weight mean|w|, std, sparsity at thresholds

Output: quantizability_analysis.txt (table) + quantizability_recommend.csv
"""
import csv
from pathlib import Path
import timm
import torch
import torch.nn as nn

OUT_DIR = Path(r'D:\xauduabo\Code+NCKH\Prune_QT_VITs\scripts')


def classify_layer(name, module):
    """Return (weight_quantize, activation_quantize, risk, reason)."""
    role = 'other'
    if 'patch_embed.proj' in name: role = 'first_conv'
    elif name == 'patch_embed.norm': role = 'first_norm'
    elif name == 'norm': role = 'final_norm'
    elif 'head' in name: role = 'classifier'
    elif 'downsample.reduction' in name: role = 'downsample'
    elif 'downsample.norm' in name: role = 'downsample_norm'
    elif '.attn.qkv' in name: role = 'attn_qkv'
    elif '.attn.proj' in name: role = 'attn_proj'
    elif '.mlp.fc1' in name: role = 'mlp_fc1'
    elif '.mlp.fc2' in name: role = 'mlp_fc2'
    elif '.norm1' in name: role = 'norm_pre_attn'
    elif '.norm2' in name: role = 'norm_pre_mlp'

    rules = {
        'first_conv':      ('NO',  'NO',  'high',   'first layer, sees raw RGB; tiny params (4.7K), no gain'),
        'first_norm':      ('NO',  'NO',  'none',   'LayerNorm gamma/beta, 192 params, sensitive'),
        'final_norm':      ('NO',  'NO',  'none',   'pre-classifier LN, critical for logits'),
        'classifier':      ('NO',  'maybe-W8', 'high', 'logits layer; quantize → top-1 drops sharply'),
        'downsample':      ('maybe-W8', 'YES', 'med', 'cross-resolution Linear, sensitive to shift'),
        'downsample_norm': ('NO',  'NO',  'none',   'LayerNorm, tiny'),
        'attn_qkv':        ('YES-APB', 'YES-channel', 'high', 'Q,K feed softmax(QK)/sqrt(d), error amplifies; V tolerant'),
        'attn_proj':       ('YES-APB', 'YES-postSoftmax', 'med', 'after attn@V; post-softmax input has heavy tail'),
        'mlp_fc1':         ('YES-APB', 'YES-postLN', 'low', 'standard FFN expansion; post-LN input well-behaved'),
        'mlp_fc2':         ('YES-APB', 'YES-postGELU', 'med', 'post-GELU input is asymmetric; needs care'),
        'norm_pre_attn':   ('NO',  'NO',  'none', 'LayerNorm scale/shift, tiny'),
        'norm_pre_mlp':    ('NO',  'NO',  'none', 'LayerNorm scale/shift, tiny'),
        'other':           ('?', '?', '?', 'unclassified'),
    }
    wq, aq, risk, reason = rules[role]
    return role, wq, aq, risk, reason


def weight_stats(m):
    if not hasattr(m, 'weight') or m.weight is None:
        return None
    w = m.weight.detach().float().flatten()
    abs_w = w.abs()
    alpha = abs_w.mean().item()
    delta = 3 * w.std().item()
    # sparsity at default APB threshold (|w| ≤ α+δ → binary)
    binary_pct = (abs_w <= (alpha + delta)).float().mean().item() * 100
    return {
        'mean_abs': alpha,
        'std': w.std().item(),
        'min': w.min().item(),
        'max': w.max().item(),
        'binary_pct_default': binary_pct,
    }


def main():
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model.eval()

    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            continue
        role, wq, aq, risk, reason = classify_layer(name, m)
        st = weight_stats(m) or {}
        rows.append({
            'name': name,
            'kind': type(m).__name__,
            'shape': tuple(m.weight.shape),
            'params': m.weight.numel() + (m.bias.numel() if getattr(m, 'bias', None) is not None else 0),
            'role': role,
            'weight_quant': wq,
            'act_quant': aq,
            'risk': risk,
            'reason': reason,
            **st,
        })

    # ---- summary by role ----
    role_summary = {}
    for r in rows:
        s = role_summary.setdefault(r['role'], {'count': 0, 'params': 0,
                                                'wq': r['weight_quant'],
                                                'risk': r['risk']})
        s['count'] += 1
        s['params'] += r['params']

    # ---- TXT output ----
    txt = OUT_DIR / 'quantizability_analysis.txt'
    with txt.open('w', encoding='utf-8') as f:
        f.write('=' * 100 + '\n')
        f.write('SWIN-S PER-LAYER QUANTIZABILITY ANALYSIS\n')
        f.write('=' * 100 + '\n\n')

        f.write('## SUMMARY BY ROLE\n\n')
        f.write(f'{"role":<18} {"count":>5} {"params":>14} {"weight_quant":<12} {"risk":<6}\n')
        f.write('-' * 70 + '\n')
        for role, s in sorted(role_summary.items(), key=lambda kv: -kv[1]['params']):
            f.write(f'{role:<18} {s["count"]:>5} {s["params"]:>14,} {s["wq"]:<12} {s["risk"]:<6}\n')

        f.write('\n\n## RECOMMENDED CONFIGURATION\n\n')
        f.write('CAN quantize (APB target, 96 Linears, 47M params, 95% of model):\n')
        f.write('  - attn.qkv  (24 layers, 11.8M)   risk=HIGH  → APB Option A (split Q/K/V) recommended\n')
        f.write('  - attn.proj (24 layers,  3.9M)   risk=MED   → APB standard\n')
        f.write('  - mlp.fc1   (24 layers, 15.7M)   risk=LOW   → APB aggressive (high binary ratio OK)\n')
        f.write('  - mlp.fc2   (24 layers, 15.7M)   risk=MED   → APB standard (post-GELU sensitive)\n\n')
        f.write('CAN quantize but skip in default config:\n')
        f.write('  - downsample.reduction (3 layers, 1.5M)  → W8 if needed, NOT APB\n')
        f.write('  - head.fc (1 layer, 0.77M)               → keep FP\n\n')
        f.write('CANNOT/SHOULD-NOT quantize:\n')
        f.write('  - patch_embed.proj (Conv2d, 4.7K)   → too small, processes raw RGB\n')
        f.write('  - All 53 LayerNorms (~70K total)    → tiny, critical for distribution stability\n\n')

        f.write('## PER-LAYER DETAIL (Linear & Conv2d only, sorted by params desc)\n\n')
        f.write(f'{"params":>9} {"shape":<22} {"role":<14} {"wq":<10} {"aq":<18} '
                f'{"risk":<5} {"αinit":>7} {"δinit":>7} {"bin%":>5}  name\n')
        f.write('-' * 160 + '\n')
        big = [r for r in rows if r['kind'] in ('Linear', 'Conv2d')]
        for r in sorted(big, key=lambda x: -x['params']):
            alpha = r.get('mean_abs', 0)
            delta = 3 * r.get('std', 0)
            bin_pct = r.get('binary_pct_default', 0)
            f.write(f'{r["params"]:>9,} {str(r["shape"]):<22} {r["role"]:<14} '
                    f'{r["weight_quant"]:<10} {r["act_quant"]:<18} {r["risk"]:<5} '
                    f'{alpha:>7.4f} {delta:>7.4f} {bin_pct:>5.1f}  {r["name"]}\n')

        f.write('\n\n## ACTIVATION QUANTIZATION POSITIONS (forward pass)\n\n')
        f.write('Per Swin block, activations to quantize (FIMA-Q does layer-wise A4):\n')
        f.write('  1. norm1.out        → input to attn.qkv      (post-LN, ~Gaussian, easy)\n')
        f.write('  2. qkv.out          → split into Q, K, V     (linear output, easy)\n')
        f.write('  3. Q @ K.T / sqrt(d) → input to softmax       (long tail, HARD)\n')
        f.write('  4. softmax.out      → input to V matmul       (in [0,1], HEAVY TAIL, HARDEST)\n')
        f.write('  5. attn @ V         → input to attn.proj      (post-softmax-weighted, MED)\n')
        f.write('  6. attn.proj.out    → residual add (KEEP FP)\n')
        f.write('  7. norm2.out        → input to mlp.fc1        (post-LN, ~Gaussian, easy)\n')
        f.write('  8. fc1.out          → input to GELU            (linear output, easy)\n')
        f.write('  9. GELU.out         → input to mlp.fc2         (post-GELU asymmetric, MED)\n')
        f.write(' 10. fc2.out          → residual add (KEEP FP)\n\n')
        f.write('Critical: residual additions and skip connections stay FP.\n')

    print(f'wrote {txt}')

    # ---- CSV ----
    csv_path = OUT_DIR / 'quantizability_recommend.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        fields = ['name', 'kind', 'shape', 'params', 'role', 'weight_quant',
                  'act_quant', 'risk', 'mean_abs', 'std', 'binary_pct_default', 'reason']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            row = dict(r)
            row['shape'] = str(row['shape'])
            w.writerow(row)
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
