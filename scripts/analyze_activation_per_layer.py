"""
Per-LAYER activation quantization analysis.

For each of the 96 APB target Linears (+ 4 skip Linears), identifies:
  - WHAT activation feeds its input (post-LN, post-GELU, post-softmax, etc.)
  - Distribution characteristics (symmetric/asymmetric, bounded, long-tail)
  - Recommended quantizer type
  - Risk level

Plus extra non-Linear activations (softmax inputs/outputs in attention).
"""
import csv
from pathlib import Path
import timm
import torch.nn as nn

OUT_DIR = Path(r'D:\xauduabo\Code+NCKH\Prune_QT_VITs\scripts')


def activation_for_linear(layer_name):
    """Return (input_source, distribution, quantizer, risk, notes)."""
    if '.attn.qkv' in layer_name:
        return ('LayerNorm(norm1).out',
                'Gaussian-like, near-zero-mean',
                'uniform-layer-A4', 'low',
                'Standard post-LN input; well-conditioned')

    if '.attn.proj' in layer_name:
        return ('attn @ V (post-softmax aggregation)',
                'mixed; depends on attention sparsity',
                'uniform-channel-A4', 'med',
                'If softmax peaks (sparse attn) → outliers; channel-wise helps')

    if '.mlp.fc1' in layer_name:
        return ('LayerNorm(norm2).out',
                'Gaussian-like, near-zero-mean',
                'uniform-layer-A4', 'low',
                'Standard post-LN input; well-conditioned')

    if '.mlp.fc2' in layer_name:
        return ('GELU.out',
                'asymmetric: tail in +, clipped at ~-0.17 in -',
                'asymmetric-uniform-A4',
                'med',
                'GELU output has nonzero mean; needs asymmetric quantizer')

    if 'downsample.reduction' in layer_name:
        return ('LayerNorm(downsample.norm).out (concatenated 2x2 patches)',
                'Gaussian-like but 4x channel concat → wider variance per channel',
                'uniform-channel-A8 (if quantized)', 'med',
                'Cross-resolution; sensitive')

    if layer_name == 'head.fc':
        return ('LayerNorm(final norm).out + global pool',
                'Gaussian, near-zero-mean',
                'KEEP FP (or A8)', 'high',
                'Last layer; quantize → logits noise → top-1 drop')

    if 'patch_embed' in layer_name:
        return ('raw RGB image (normalized)',
                'fixed by ImageNet normalize',
                'KEEP FP', 'high',
                'Input layer; never quantized')

    return ('?', '?', '?', '?', 'unknown')


def main():
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=False)

    # 1) Per-Linear activation table (100 Linears total: 96 APB + 3 downsample + 1 head)
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        src, dist, qntz, risk, notes = activation_for_linear(name)
        rows.append({
            'name': name,
            'shape': tuple(m.weight.shape),
            'in_dim': m.in_features,
            'out_dim': m.out_features,
            'act_input_source': src,
            'distribution': dist,
            'quantizer': qntz,
            'risk': risk,
            'notes': notes,
        })

    # 2) Non-Linear activations (matmul-only positions inside attention)
    attn_extras = [
        {
            'name': '<block>.attn.q_at_kT',
            'shape': '(B*nW, heads, N, N)',
            'in_dim': None,
            'out_dim': None,
            'act_input_source': 'Q @ K.T / sqrt(d_head)',
            'distribution': 'long-tail, mostly small + few large peaks',
            'quantizer': 'log2 / power-of-2',
            'risk': 'high',
            'notes': 'feeds softmax; uniform quantization clips large peaks',
        },
        {
            'name': '<block>.attn.softmax_out',
            'shape': '(B*nW, heads, N, N)',
            'in_dim': None,
            'out_dim': None,
            'act_input_source': 'softmax(QK^T + rel_pos_bias)',
            'distribution': '[0,1] heavy-tail (most ≈0, few near 1)',
            'quantizer': 'log2 (post-softmax specialized)',
            'risk': 'high',
            'notes': 'attention sparsity → most values tiny; linear quantizer wastes range',
        },
    ]

    # ---- TXT output ----
    txt = OUT_DIR / 'activation_per_layer.txt'
    with txt.open('w', encoding='utf-8') as f:
        f.write('=' * 110 + '\n')
        f.write('PER-LAYER ACTIVATION QUANTIZATION ANALYSIS (Swin-S)\n')
        f.write('=' * 110 + '\n\n')

        f.write('## SUMMARY: 100 Linear input activations + 2 attention matmul activations per block\n\n')

        f.write('Group by source/distribution:\n\n')
        groups = {}
        for r in rows:
            key = (r['act_input_source'], r['distribution'], r['quantizer'], r['risk'])
            groups.setdefault(key, []).append(r['name'])
        for (src, dist, qntz, risk), names in groups.items():
            f.write(f'  [{len(names)}x] source={src}\n')
            f.write(f'       distribution: {dist}\n')
            f.write(f'       quantizer:    {qntz}\n')
            f.write(f'       risk:         {risk}\n')
            f.write(f'       example:      {names[0]}\n\n')

        f.write('\n## PER-LINEAR DETAIL\n\n')
        f.write(f'{"layer":<45} {"in_dim":>6} → {"out":>5}  {"quantizer":<22} '
                f'{"risk":<5}  source\n')
        f.write('-' * 160 + '\n')
        for r in rows:
            f.write(f'{r["name"]:<45} {r["in_dim"]:>6} → {r["out_dim"]:>5}  '
                    f'{r["quantizer"]:<22} {r["risk"]:<5}  {r["act_input_source"]}\n')

        f.write('\n\n## NON-LINEAR ACTIVATIONS (matmul inside attention)\n\n')
        f.write('These are NOT inputs to a Linear, but are quantized in W4/A4 setting:\n\n')
        for r in attn_extras:
            f.write(f'  - {r["name"]}\n')
            f.write(f'      source:       {r["act_input_source"]}\n')
            f.write(f'      distribution: {r["distribution"]}\n')
            f.write(f'      quantizer:    {r["quantizer"]}\n')
            f.write(f'      risk:         {r["risk"]}\n')
            f.write(f'      notes:        {r["notes"]}\n\n')

        f.write('\n## QUANTIZER COUNT REQUIRED (per block × 24 blocks)\n\n')
        f.write('Per Swin block, FIMA-Q standard setup needs:\n')
        f.write('  4 input quantizers for Linears (qkv, proj, fc1, fc2)\n')
        f.write('  2 matmul quantizers in attention (QK^T, softmax_out)\n')
        f.write('  = 6 activation quantizers per block × 24 blocks = 144 quantizers\n\n')
        f.write('+ 3 input quantizers for downsample.reduction (stages 1,2,3)\n')
        f.write('+ 1 input quantizer for head.fc (optional, often FP)\n')
        f.write('= ~148 activation quantizers total in full Swin-S\n\n')

        f.write('\n## DECISION TABLE: which to quantize aggressively vs keep high precision\n\n')
        f.write(f'{"level":<8} {"count":>5}  description\n')
        f.write('-' * 90 + '\n')
        f.write('A4-easy    48   inputs to qkv + fc1 (post-LN, well-conditioned)\n')
        f.write('A4-asym    24   inputs to fc2 (post-GELU, asymmetric)\n')
        f.write('A4-chan    24   inputs to attn.proj (post softmax@V)\n')
        f.write('A4-log     48   QK^T inputs + softmax outputs (heavy-tail)\n')
        f.write('A8-or-FP    4   downsample inputs + head input\n')

    print(f'wrote {txt}')

    # ---- CSV ----
    csv_path = OUT_DIR / 'activation_per_layer.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        fields = ['name', 'shape', 'in_dim', 'out_dim', 'act_input_source',
                  'distribution', 'quantizer', 'risk', 'notes']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows + attn_extras:
            row = dict(r)
            row['shape'] = str(row['shape'])
            w.writerow(row)
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
