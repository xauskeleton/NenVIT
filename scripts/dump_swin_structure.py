"""
Dump Swin-S structure to 2 files:
  - swin_structure_full.txt   : full torch print(model), verbose
  - swin_structure_compact.txt: concise grouped summary by layer pattern
"""
from pathlib import Path
import timm
import torch.nn as nn

OUT_DIR = Path(r'D:\xauduabo\Code+NCKH\Prune_QT_VITs\scripts')


def dump_full(model, path: Path):
    with path.open('w', encoding='utf-8') as f:
        f.write(str(model))
    print(f'[full]    wrote {path}  ({path.stat().st_size:,} bytes)')


def dump_compact(model, path: Path):
    lines = []
    lines.append('=' * 80)
    lines.append('SWIN-S COMPACT STRUCTURE (grouped by layer pattern)')
    lines.append('=' * 80)

    # 1. Top-level overview
    lines.append('\n## TOP-LEVEL\n')
    for name, child in model.named_children():
        lines.append(f'  {name:<20} {type(child).__name__}')

    # 2. Per-stage block count + channel
    lines.append('\n## STAGES (4 stages, depths follow Swin-S spec)\n')
    for stage_idx in range(4):
        stage = model.layers[stage_idx]
        blocks = stage.blocks if hasattr(stage, 'blocks') else []
        depth = len(blocks)
        if depth > 0:
            blk0 = blocks[0]
            qkv = blk0.attn.qkv
            C = qkv.in_features
            lines.append(f'  stage_{stage_idx}: depth={depth:>2}  C={C:>3}  '
                         f'qkv={tuple(qkv.weight.shape)}')

    # 3. Linear groups (collapsed)
    lines.append('\n## LINEAR LAYERS (grouped by leaf name)\n')
    groups = {}  # leaf_name -> [(name, shape, params)]
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            leaf = name.split('.')[-1]
            shape = tuple(m.weight.shape)
            params = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            groups.setdefault(leaf, []).append((name, shape, params))

    total_linear = 0
    for leaf, items in groups.items():
        n_inst = len(items)
        unique_shapes = sorted(set(it[1] for it in items))
        sum_params = sum(it[2] for it in items)
        total_linear += sum_params
        lines.append(f'\n  [{leaf}]  {n_inst} instances, total {sum_params:,} params '
                     f'({sum_params/1e6:.2f}M)')
        for shape in unique_shapes:
            matches = [it for it in items if it[1] == shape]
            lines.append(f'    weight={str(shape):<20}  x{len(matches):<3}  '
                         f'first: {matches[0][0]}')

    # 4. Non-linear leaves
    lines.append('\n## NON-LINEAR LAYERS\n')
    conv_count = ln_count = drop_count = act_count = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_count += 1
            lines.append(f'  Conv2d   {name}  weight={tuple(m.weight.shape)}')
        elif isinstance(m, nn.LayerNorm):
            ln_count += 1
        elif isinstance(m, (nn.Dropout, nn.GELU, nn.ReLU)):
            if isinstance(m, nn.Dropout): drop_count += 1
            else: act_count += 1
    lines.append(f'  Conv2d   total: {conv_count}')
    lines.append(f'  LayerNorm total: {ln_count}')
    lines.append(f'  Activation/Dropout total: {act_count + drop_count}')

    # 5. APB target/skip classification
    lines.append('\n## APB TARGETS (filter logic)\n')
    skip_patterns = ['downsample.reduction', 'head', 'patch_embed']
    apb_targets = []
    apb_skip = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(p in name for p in skip_patterns):
            apb_skip.append((name, tuple(m.weight.shape)))
        else:
            apb_targets.append((name, tuple(m.weight.shape)))
    target_params = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
                        for n, m in model.named_modules()
                        if isinstance(m, nn.Linear) and not any(p in n for p in skip_patterns))
    lines.append(f'  APB target Linears:  {len(apb_targets):>3}  '
                 f'(~{target_params/1e6:.2f}M params)')
    lines.append(f'  APB skip Linears:    {len(apb_skip):>3}')
    lines.append('  Skip list:')
    for name, shape in apb_skip:
        lines.append(f'    SKIP  {name}  {shape}')

    # 6. Param summary
    total = sum(p.numel() for p in model.parameters())
    lines.append('\n## PARAM TOTALS\n')
    lines.append(f'  Total params:        {total:>15,}  ({total/1e6:.2f}M)')
    lines.append(f'  Linear params:       {total_linear:>15,}  ({100*total_linear/total:.1f}%)')
    lines.append(f'  APB-eligible params: {target_params:>15,}  ({100*target_params/total:.1f}%)')

    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[compact] wrote {path}  ({path.stat().st_size:,} bytes)')


def main():
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=False)
    dump_full(model, OUT_DIR / 'swin_structure_full.txt')
    dump_compact(model, OUT_DIR / 'swin_structure_compact.txt')


if __name__ == '__main__':
    main()
