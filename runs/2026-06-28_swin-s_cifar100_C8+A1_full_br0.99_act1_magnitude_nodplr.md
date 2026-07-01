# Swin-S APB W+A — CIFAR-100 (C8+A1 **MAGNITUDE / no-DPLR** / full / br0.99 / act 1-bit)

Run date: 2026-06-28 (~7.5h) | Script: `apb_fimaq/qat.py` | `--partition magnitude`, KHÔNG `--use-dplr-loss`
**APB thuần nhất:** partition theo |w|, CE-only (không FIM, không distillation). Cell (magnitude, no-DPLR) của 2×2.

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full --epochs 29 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 1 --partition magnitude
```
(FIM bị bỏ qua hoàn toàn: `Skipping FIM extraction (partition='magnitude', no DPLR loss)`.)

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 0.94% | 4.94% |
| **Best (ep27)** | **48.87%** | 78.57% |

- quant cost vs FP 90.88% = **−42.01%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~930s/epoch × 29 ≈ **7.5h**. val plateau ~ep25-29 (47.8→48.9).

## ⭐ Ma trận 2×2 (partition × DPLR) — act-1 / full / br0.99 / 29ep / batch32 / seed3407
| | no-DPLR | DPLR |
|-----------|---------|------|
| **fim** | _(Test 3 chưa chạy)_ | **51.03%** |
| **magnitude** | **48.87%** (file này) | **58.29%** |

**Đọc được tới giờ:**
- **DPLR giúp magnitude RẤT nhiều: +9.42%** (48.87 → 58.29). Distillation là yếu tố lớn ở chế độ cực hạn này.
- **magnitude > fim +7.26%** (58.29 vs 51.03, cùng DPLR) vẫn đứng — ablation sạch.
- Còn thiếu **fim no-DPLR (Test 3)** để biết DPLR giúp fim bao nhiêu, và so magnitude-vs-fim khi KHÔNG DPLR.
  Nếu Test 3 ≈ 50% → DPLR giúp fim ít (~1%) nhưng giúp magnitude nhiều (~9%) → tương tác thú vị.

## ⚡ Tốc độ: no-DPLR nhanh hơn hẳn
~930s/ep (1.86 it/s) vs DPLR ~1128s/ep (1.50 it/s) → no-DPLR nhanh ~18% (không chạy forward teacher).
Tổng 7.5h vs 9.1h.

## Per-epoch (đầy đủ) — CE-only nên không có cột ce/dplr
```
Ep  1: train_loss=4.1884 | top1=9.77%  top5=31.00% | 932.9s
Ep  2: train_loss=3.6582 | top1=15.16% top5=39.86% | 934.1s
Ep  3: train_loss=3.3620 | top1=21.07% top5=49.59% | 934.2s
Ep  4: train_loss=3.1362 | top1=23.42% top5=51.54% | 934.7s
Ep  5: train_loss=2.9688 | top1=26.60% top5=57.51% | 933.1s
Ep  6: train_loss=2.8258 | top1=29.88% top5=60.26% | 933.6s
Ep  7: train_loss=2.6951 | top1=30.95% top5=61.35% | 933.6s
Ep  8: train_loss=2.5918 | top1=32.64% top5=63.20% | 932.8s
Ep  9: train_loss=2.4862 | top1=34.56% top5=65.98% | 932.3s
Ep 10: train_loss=2.3980 | top1=35.24% top5=67.22% | 927.2s
Ep 11: train_loss=2.3118 | top1=37.35% top5=68.54% | 929.4s
Ep 12: train_loss=2.2295 | top1=39.39% top5=70.46% | 930.3s
Ep 13: train_loss=2.1580 | top1=40.74% top5=71.48% | 930.4s
Ep 14: train_loss=2.0890 | top1=39.61% top5=71.91% | 930.6s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: train_loss=2.0264 | top1=42.34% top5=73.30% | 928.6s
Ep 16: train_loss=1.9588 | top1=43.35% top5=74.23% | 930.8s
Ep 17: train_loss=1.9069 | top1=43.71% top5=74.88% | 928.6s
Ep 18: train_loss=1.8458 | top1=44.24% top5=75.81% | 927.9s
Ep 19: train_loss=1.8026 | top1=45.74% top5=76.01% | 927.6s
Ep 20: train_loss=1.7630 | top1=46.00% top5=76.03% | 927.7s
Ep 21: train_loss=1.7110 | top1=46.64% top5=76.56% | 930.3s
Ep 22: train_loss=1.6822 | top1=47.30% top5=77.31% | 930.8s
Ep 23: train_loss=1.6420 | top1=47.46% top5=77.28% | 927.8s
Ep 24: train_loss=1.6050 | top1=47.53% top5=77.71% | 927.7s
Ep 25: train_loss=1.5834 | top1=47.75% top5=77.91% | 930.3s
Ep 26: train_loss=1.5520 | top1=48.67% top5=78.15% | 928.3s
Ep 27: train_loss=1.5418 | top1=48.87% top5=78.57% | 931.6s   ← BEST
Ep 28: train_loss=1.5253 | top1=48.62% top5=78.38% | 927.9s
Ep 29: train_loss=1.5156 | top1=48.80% top5=78.58% | 927.4s
```

## TODO
- **Test 3 = fim + no-DPLR** (`--partition fim`, bỏ `--use-dplr-loss`) → lấp ô cuối của 2×2.
- Sau khi đủ 2×2: thử br0.95 / act≥2 xem FIM có lật lại magnitude khi bớt cực hạn.
- ⚠️ single seed (3407).
