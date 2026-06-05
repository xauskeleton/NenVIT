# Swin-S FIMA-Q structural pruning — CIFAR-100 (P4: per_layer, head0.7 / mlp0.7)

Run date: 2026-06-02 14:21 | Script: `apb_fimaq/prune_swin_cifar.py`
Init: CIFAR baseline `ckpt/best.pth` → `[FP baseline] top1=90.88% top5=98.97%`

## Command
```bash
python apb_fimaq/prune_swin_cifar.py --rank-mode per_layer \
  --head-ratio 0.7 --mlp-ratio 0.7 --min-heads 1 \
  --fim-batches 10 --epochs 20 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407
```

## Summary
| Stage | top-1 | top-5 | params |
| ----- | ----- | ----- | ------ |
| FP baseline | 90.88% | 98.97% | 48.91M |
| Post-prune (trước FT) | 1.30% | 6.46% | 16.19M |
| **Best (ep20)** | **87.85%** | 97.85% | 16.19M |

- **prune cost = −3.03%** | params 48.91M → **16.19M (66.9% nhỏ hơn)**
- heads 282→92, mlp 36096→10832 | FIM 17.4s

## Per-epoch
```
Ep  1: loss=2.2135 | top1=77.48% top5=95.71% | 322.0s
Ep  2: loss=1.3859 | top1=80.53% top5=96.78% | 321.5s
Ep  3: loss=1.2470 | top1=82.86% top5=97.45% | 321.6s
Ep  4: loss=1.1618 | top1=83.25% top5=97.47% | 321.3s
Ep  5: loss=1.0970 | top1=83.86% top5=97.58% | 321.9s
Ep  6: loss=1.0538 | top1=84.68% top5=97.69% | 322.2s
Ep  7: loss=1.0136 | top1=84.86% top5=97.75% | 321.5s
Ep  8: loss=0.9838 | top1=85.53% top5=97.83% | 321.8s
Ep  9: loss=0.9534 | top1=86.06% top5=98.01% | 321.6s
Ep 10: loss=0.9325 | top1=85.97% top5=98.00% | 321.5s
Ep 11: loss=0.9115 | top1=86.57% top5=98.00% | 321.6s
Ep 12: loss=0.8945 | top1=86.82% top5=97.89% | 321.6s
Ep 13: loss=0.8801 | top1=86.87% top5=97.97% | 322.0s
Ep 14: loss=0.8658 | top1=87.01% top5=97.95% | 321.6s
Ep 15: loss=0.8581 | top1=87.34% top5=97.91% | 321.4s
Ep 16: loss=0.8508 | top1=87.53% top5=97.93% | 321.5s
Ep 17: loss=0.8445 | top1=87.85% top5=97.93% | 322.2s   ← BEST
Ep 18: loss=0.8422 | top1=87.49% top5=97.93% | 321.9s
Ep 19: loss=0.8375 | top1=87.65% top5=97.90% | 321.3s
Ep 20: loss=0.8358 | top1=87.76% top5=97.85% | 321.8s
```

## Note
≈ ngang P3 (global 0.7: 87.80% @ 14.27M) về acc nhưng tốn thêm ~2M params → global hiệu quả hơn.
