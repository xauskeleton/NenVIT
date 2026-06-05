# Swin-S FIMA-Q structural pruning — CIFAR-100 (P2: per_layer, head0.5 / mlp0.5)

Run date: 2026-06-02 14:19 | Script: `apb_fimaq/prune_swin_cifar.py`
Init: CIFAR baseline `ckpt/best.pth` → `[FP baseline] top1=90.88% top5=98.97%`
**So sánh với P1** (global, cùng 0.5): xem `runs/README.md` bảng global vs per_layer.

## Command
```bash
python apb_fimaq/prune_swin_cifar.py --rank-mode per_layer \
  --head-ratio 0.5 --mlp-ratio 0.5 --min-heads 1 \
  --fim-batches 10 --epochs 20 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407
```

## Summary
| Stage | top-1 | top-5 | params |
| ----- | ----- | ----- | ------ |
| FP baseline | 90.88% | 98.97% | 48.91M |
| Post-prune (trước FT) | 7.01% | 22.51% | 25.32M |
| **Best (ep20)** | **90.55%** | 98.51% | 25.32M |

- **prune cost = −0.33%** | params 48.91M → **25.32M (48.2% nhỏ hơn)**
- heads 282→142, mlp 36096→18048 | FIM 20.1s, 96 layers

## Per-epoch
```
Ep  1: loss=1.2529 | top1=86.11% top5=98.16% | 428.1s
Ep  2: loss=1.0216 | top1=87.17% top5=98.40% | 426.9s
Ep  3: loss=0.9700 | top1=87.56% top5=98.27% | 426.5s
Ep  4: loss=0.9375 | top1=87.73% top5=98.39% | 426.4s
Ep  5: loss=0.9108 | top1=88.23% top5=98.38% | 427.9s
Ep  6: loss=0.8954 | top1=88.71% top5=98.55% | 427.7s
Ep  7: loss=0.8772 | top1=88.47% top5=98.59% | 426.7s
Ep  8: loss=0.8635 | top1=89.09% top5=98.51% | 427.2s
Ep  9: loss=0.8509 | top1=89.13% top5=98.53% | 428.4s
Ep 10: loss=0.8392 | top1=89.44% top5=98.44% | 427.2s
Ep 11: loss=0.8300 | top1=89.81% top5=98.39% | 424.8s
Ep 12: loss=0.8248 | top1=89.83% top5=98.52% | 428.4s
Ep 13: loss=0.8152 | top1=89.85% top5=98.54% | 426.4s
Ep 14: loss=0.8090 | top1=90.16% top5=98.67% | 427.8s
Ep 15: loss=0.8060 | top1=90.30% top5=98.49% | 427.5s
Ep 16: loss=0.8037 | top1=90.42% top5=98.58% | 426.8s
Ep 17: loss=0.8005 | top1=90.26% top5=98.53% | 428.2s
Ep 18: loss=0.7991 | top1=90.50% top5=98.49% | 426.9s
Ep 19: loss=0.7981 | top1=90.54% top5=98.51% | 426.4s
Ep 20: loss=0.7968 | top1=90.55% top5=98.51% | 428.2s   ← BEST
```

## Note
per_layer 0.5 thua P1 (global 0.5: 90.87% @ 23.06M): kém 0.32% acc VÀ tốn thêm 2.3M params → global dominate.
