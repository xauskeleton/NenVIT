# Swin-S FIMA-Q structural pruning — CIFAR-100 (P3: global / per_param, head0.7 / mlp0.7)

Run date: 2026-06-02 14:30 | Script: `apb_fimaq/prune_swin_cifar.py`
Init: CIFAR baseline `ckpt/best.pth` → `[FP baseline] top1=90.88% top5=98.97%`

## Command
```bash
python apb_fimaq/prune_swin_cifar.py --rank-mode global --global-metric per_param \
  --head-ratio 0.7 --mlp-ratio 0.7 --min-heads 1 --mlp-keep-frac 0.05 \
  --fim-batches 10 --epochs 20 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407
```

## Summary
| Stage | top-1 | top-5 | params |
| ----- | ----- | ----- | ------ |
| FP baseline | 90.88% | 98.97% | 48.91M |
| Post-prune (trước FT) | 2.05% | 7.71% | 14.27M |
| **Best (ep20)** | **87.80%** | 98.16% | 14.27M |

- **prune cost = −3.08%** | params 48.91M → **14.27M (70.8% nhỏ hơn)**
- heads 282→85, mlp 36096→10829 | FIM 18.2s

## Per-epoch
```
Ep  1: loss=2.0161 | top1=78.15% top5=96.14% | 378.7s
Ep  2: loss=1.3739 | top1=80.75% top5=96.95% | 378.2s
Ep  3: loss=1.2410 | top1=83.25% top5=97.58% | 378.2s
Ep  4: loss=1.1596 | top1=83.87% top5=97.69% | 378.2s
Ep  5: loss=1.1016 | top1=84.76% top5=97.93% | 378.1s
Ep  6: loss=1.0550 | top1=84.80% top5=97.97% | 377.8s
Ep  7: loss=1.0162 | top1=85.10% top5=97.88% | 377.9s
Ep  8: loss=0.9856 | top1=85.65% top5=97.93% | 378.3s
Ep  9: loss=0.9601 | top1=85.79% top5=98.08% | 378.1s
Ep 10: loss=0.9320 | top1=86.20% top5=97.96% | 378.6s
Ep 11: loss=0.9138 | top1=86.45% top5=98.02% | 377.9s
Ep 12: loss=0.8972 | top1=86.97% top5=98.17% | 377.8s
Ep 13: loss=0.8838 | top1=86.98% top5=97.95% | 377.7s
Ep 14: loss=0.8694 | top1=87.23% top5=98.15% | 378.0s
Ep 15: loss=0.8601 | top1=87.69% top5=98.06% | 378.2s
Ep 16: loss=0.8539 | top1=87.39% top5=98.10% | 378.2s
Ep 17: loss=0.8478 | top1=87.68% top5=98.09% | 378.1s
Ep 18: loss=0.8437 | top1=87.75% top5=98.16% | 378.6s
Ep 19: loss=0.8406 | top1=87.80% top5=98.15% | 378.5s   ← BEST
Ep 20: loss=0.8394 | top1=87.80% top5=98.16% | 378.1s
```

## Note
Sweet-spot nén cao: −70.8% params chỉ mất −3.08% acc. So per_layer 0.7 (P4: 87.85% @ 16.19M):
acc ~hòa nhưng global ít hơn ~2M params (−12%) → global Pareto-tốt hơn. Pareto frontier fair.
