# Swin-S FIMA-Q structural pruning — CIFAR-100 (P5: global / per_param, head0.9 / mlp0.9)

Run date: 2026-06-02 14:29 | Script: `apb_fimaq/prune_swin_cifar.py`
Init: CIFAR baseline `ckpt/best.pth` → `[FP baseline] top1=90.88% top5=98.97%`

## Command
```bash
python apb_fimaq/prune_swin_cifar.py --rank-mode global --global-metric per_param \
  --head-ratio 0.9 --mlp-ratio 0.9 --min-heads 1 --mlp-keep-frac 0.05 \
  --fim-batches 10 --epochs 20 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407
```

## Summary
| Stage | top-1 | top-5 | params |
| ----- | ----- | ----- | ------ |
| FP baseline | 90.88% | 98.97% | 48.91M |
| Post-prune (trước FT) | 1.03% | 5.25% | 5.97M |
| **Best (ep18)** | **63.00%** | 89.31% | 5.97M |

- **prune cost = −27.88%** | params 48.91M → **5.97M (87.8% nhỏ hơn)** — model nhỏ nhất
- heads 282→28, mlp 36096→3610 | FIM 16.0s

## Per-epoch
```
Ep  1: loss=3.8113 | top1=26.05% top5=57.58% | 275.0s
Ep  2: loss=3.0313 | top1=37.53% top5=70.73% | 277.9s
Ep  3: loss=2.7224 | top1=42.59% top5=75.73% | 278.4s
Ep  4: loss=2.5277 | top1=46.72% top5=79.30% | 281.7s
Ep  5: loss=2.3879 | top1=50.17% top5=81.04% | 288.3s
Ep  6: loss=2.2824 | top1=52.49% top5=83.03% | 288.2s
Ep  7: loss=2.1878 | top1=54.15% top5=84.16% | 286.3s
Ep  8: loss=2.1114 | top1=56.08% top5=85.29% | 279.3s
Ep  9: loss=2.0441 | top1=57.54% top5=86.29% | 278.5s
Ep 10: loss=1.9840 | top1=58.19% top5=86.75% | 278.7s
Ep 11: loss=1.9315 | top1=59.20% top5=87.67% | 278.6s
Ep 12: loss=1.8890 | top1=60.38% top5=87.47% | 278.5s
Ep 13: loss=1.8422 | top1=60.85% top5=88.03% | 278.5s
Ep 14: loss=1.8117 | top1=61.34% top5=88.27% | 278.6s
Ep 15: loss=1.7823 | top1=61.74% top5=88.65% | 278.9s
Ep 16: loss=1.7570 | top1=62.30% top5=88.72% | 278.3s
Ep 17: loss=1.7358 | top1=62.45% top5=88.99% | 282.0s
Ep 18: loss=1.7232 | top1=63.00% top5=89.31% | 288.1s   ← BEST
Ep 19: loss=1.7133 | top1=62.87% top5=89.18% | 288.0s
Ep 20: loss=1.7091 | top1=62.87% top5=89.19% | 284.9s
```

## Note
⚠️ Cùng đỉnh nén (~6M params) nhưng global thắng per_layer 0.9 (P6: 60.64% @ 6.20M): +2.36% acc,
ít params hơn. Tuy vậy 20 ep vẫn chưa hội tụ hẳn (val gần plateau ep18–20). Cost −27.88% quá đắt
→ chỉ dùng khi cực kỳ ưu tiên storage; sweet spot vẫn là 0.5–0.7.
