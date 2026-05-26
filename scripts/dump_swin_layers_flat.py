"""
Dump EVERY layer (Linear/Conv2d/LayerNorm) one per line for layer-level work.
Output: swin_layers_flat.txt + swin_layers_flat.csv
"""
import csv
from pathlib import Path
import timm
import torch.nn as nn

OUT_DIR = Path(r'D:\xauduabo\Code+NCKH\Prune_QT_VITs\scripts')
SKIP_PATTERNS = ['downsample.reduction', 'head', 'patch_embed']


def kind_of(module):
    if isinstance(module, nn.Linear): return 'Linear'
    if isinstance(module, nn.Conv2d): return 'Conv2d'
    if isinstance(module, nn.LayerNorm): return 'LayerNorm'
    return type(module).__name__


def apb_decision(name, module):
    if not isinstance(module, nn.Linear):
        return 'SKIP_NON_LINEAR'
    if any(p in name for p in SKIP_PATTERNS):
        return 'SKIP_BY_RULE'
    return 'APB_TARGET'


def role_of(name):
    # block_idx parsing
    if 'patch_embed' in name: return 'first'
    if name == 'norm' or name == 'head.fc' or 'head' in name: return 'last'
    if 'downsample.reduction' in name: return 'downsample'
    if '.attn.qkv' in name: return 'attn_qkv'
    if '.attn.proj' in name: return 'attn_proj'
    if '.mlp.fc1' in name: return 'mlp_fc1'
    if '.mlp.fc2' in name: return 'mlp_fc2'
    if 'norm' in name: return 'norm'
    return 'other'


def stage_of(name):
    if name.startswith('layers.'):
        return int(name.split('.')[1])
    return -1


def block_of(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'blocks' and i + 1 < len(parts):
            return int(parts[i + 1])
    return -1


def main():
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=False)

    rows = []  # list of dicts
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            shape = tuple(m.weight.shape)
            params = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            rows.append({
                'idx': len(rows),
                'kind': kind_of(m),
                'name': name,
                'shape': shape,
                'params': params,
                'stage': stage_of(name),
                'block': block_of(name),
                'role': role_of(name),
                'apb': apb_decision(name, m),
            })

    # --- TXT (human-readable) ---
    txt_path = OUT_DIR / 'swin_layers_flat.txt'
    with txt_path.open('w', encoding='utf-8') as f:
        f.write(f'{"idx":>4} {"kind":<9} {"stg":>3} {"blk":>3} {"role":<12} '
                f'{"params":>10} {"shape":<22} {"apb":<18} name\n')
        f.write('-' * 130 + '\n')
        for r in rows:
            f.write(f'{r["idx"]:>4} {r["kind"]:<9} {r["stage"]:>3} {r["block"]:>3} '
                    f'{r["role"]:<12} {r["params"]:>10,} '
                    f'{str(r["shape"]):<22} {r["apb"]:<18} {r["name"]}\n')

        # totals
        n_target = sum(1 for r in rows if r['apb'] == 'APB_TARGET')
        n_skip_rule = sum(1 for r in rows if r['apb'] == 'SKIP_BY_RULE')
        n_skip_nonlin = sum(1 for r in rows if r['apb'] == 'SKIP_NON_LINEAR')
        p_target = sum(r['params'] for r in rows if r['apb'] == 'APB_TARGET')
        p_total = sum(r['params'] for r in rows)
        f.write('-' * 130 + '\n')
        f.write(f'TOTAL: {len(rows)} layers | '
                f'APB_TARGET={n_target} | SKIP_BY_RULE={n_skip_rule} | '
                f'SKIP_NON_LINEAR={n_skip_nonlin}\n')
        f.write(f'PARAMS: target={p_target:,} ({100*p_target/p_total:.1f}%)  '
                f'total={p_total:,}\n')

    print(f'wrote {txt_path}')

    # --- CSV (for spreadsheet/programmatic) ---
    csv_path = OUT_DIR / 'swin_layers_flat.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx', 'kind', 'stage', 'block', 'role',
                                          'params', 'shape', 'apb', 'name'])
        w.writeheader()
        for r in rows:
            r['shape'] = str(r['shape'])
            w.writerow(r)
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
