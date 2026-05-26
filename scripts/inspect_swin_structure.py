"""Inspect Swin-S structure from timm: print all named modules with shapes."""
import timm
import torch.nn as nn


def main():
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=False)

    print('=' * 80)
    print('SWIN-S TOP-LEVEL STRUCTURE')
    print('=' * 80)
    for name, child in model.named_children():
        print(f'{name}: {type(child).__name__}')

    print()
    print('=' * 80)
    print('ALL LINEAR LAYERS (these are APB candidates)')
    print('=' * 80)
    total_params = 0
    linear_params = 0
    linear_groups = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            shape = tuple(m.weight.shape)
            n = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            linear_params += n
            short = name.split('.')[-1]
            linear_groups.setdefault(short, []).append((name, shape, n))

    for short, items in linear_groups.items():
        print(f'\n--- "{short}" ({len(items)} instances) ---')
        for name, shape, n in items[:3]:
            print(f'  {name}  weight={shape}  params={n:,}')
        if len(items) > 3:
            print(f'  ... ({len(items)-3} more like this)')

    print()
    print('=' * 80)
    print('CONV2D LAYERS')
    print('=' * 80)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f'  {name}  weight={tuple(m.weight.shape)}  '
                  f'stride={m.stride}  kernel={m.kernel_size}')

    print()
    print('=' * 80)
    print('LAYERNORM LAYERS')
    print('=' * 80)
    ln_count = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            ln_count += 1
    print(f'  Total LayerNorm: {ln_count}')

    print()
    print('=' * 80)
    print('PARAMETER BREAKDOWN')
    print('=' * 80)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params:           {total_params:>15,} ({total_params/1e6:.2f}M)')
    print(f'Linear params:          {linear_params:>15,} ({100*linear_params/total_params:.1f}%)')

    # Per-stage breakdown
    print()
    print('=' * 80)
    print('PER-STAGE PARAMETER COUNT (Linear layers only)')
    print('=' * 80)
    stage_params = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            parts = name.split('.')
            if len(parts) >= 2 and parts[0] == 'layers':
                stage = f'stage_{parts[1]}'
            elif 'head' in name:
                stage = 'head'
            else:
                stage = 'other'
            n = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            stage_params[stage] = stage_params.get(stage, 0) + n
    for stage, n in sorted(stage_params.items()):
        print(f'  {stage}: {n:>12,}')


if __name__ == '__main__':
    main()
