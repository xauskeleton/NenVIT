# Swin-S FIMA-Q structural pruning — CIFAR-100 (P6: per_layer, head0.9 / mlp0.9)

Run date: 2026-06-02 14:25 | Script: `apb_fimaq/prune_swin_cifar.py`
Init: CIFAR baseline `ckpt/best.pth` → `[FP baseline] top1=90.88% top5=98.97%`

## Command
```bash
python apb_fimaq/prune_swin_cifar.py --rank-mode per_layer \
  --head-ratio 0.9 --mlp-ratio 0.9 --min-heads 1 \
  --fim-batches 10 --epochs 20 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407
```

## Summary
| Stage | top-1 | top-5 | params |
| ----- | ----- | ----- | ------ |
| FP baseline | 90.88% | 98.97% | 48.91M |
| Post-prune (trước FT) | 1.00% | 5.40% | 6.20M |
| **Best (ep20)** | **60.64%** | 87.88% | 6.20M |

- **prune cost = −30.24%** | params 48.91M → **6.20M (87.3% nhỏ hơn)**
- heads 282→26, mlp 36096→3616 | FIM 19.2s

## Per-epoch
```
Ep  1: loss=3.9837 | top1=20.95% top5=49.70% | 249.2s
Ep  2: loss=3.2482 | top1=31.94% top5=64.07% | 249.1s
Ep  3: loss=2.8874 | top1=39.01% top5=71.80% | 249.2s
Ep  4: loss=2.6727 | top1=43.60% top5=75.65% | 248.8s
Ep  5: loss=2.5205 | top1=46.69% top5=78.18% | 249.0s
Ep  6: loss=2.3991 | top1=48.67% top5=80.11% | 248.8s
Ep  7: loss=2.3018 | top1=50.96% top5=81.87% | 248.9s
Ep  8: loss=2.2221 | top1=53.33% top5=83.58% | 249.5s
Ep  9: loss=2.1500 | top1=54.49% top5=84.07% | 248.8s
Ep 10: loss=2.0890 | top1=55.66% top5=84.83% | 249.7s
Ep 11: loss=2.0385 | top1=56.14% top5=84.74% | 248.4s
Ep 12: loss=1.9886 | top1=57.68% top5=85.90% | 249.4s
Ep 13: loss=1.9419 | top1=58.44% top5=86.37% | 248.8s
Ep 14: loss=1.9049 | top1=59.25% top5=86.97% | 249.3s
Ep 15: loss=1.8778 | top1=59.07% top5=87.19% | 249.5s
Ep 16: loss=1.8539 | top1=60.03% top5=87.44% | 249.3s
Ep 17: loss=1.8330 | top1=60.53% top5=87.69% | 248.8s
Ep 18: loss=1.8169 | top1=60.51% top5=87.78% | 249.4s
Ep 19: loss=1.8091 | top1=60.62% top5=87.83% | 248.8s
Ep 20: loss=1.8060 | top1=60.64% top5=87.88% | 249.0s   ← BEST (vẫn đang tăng!)
```

## Note
⚠️ 0.9 quá mạnh: post-prune ~1% (≈ random), 20 ep KHÔNG đủ recover — val vẫn tăng đều ở ep20.
Thua P5 (global 0.9: 63.00% @ 5.97M) cả acc lẫn params. Capacity floor: chỉ còn 26 heads / 3.6k MLP.
