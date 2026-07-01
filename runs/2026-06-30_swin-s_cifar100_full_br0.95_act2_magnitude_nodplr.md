# Swin-S APB W+A — CIFAR-100 (br0.95 / full / act 2-bit / **magnitude / no-DPLR**)

Run date: 2026-06-30 (~9.9h) | Script: `apb_fimaq/qat.py` | `--partition magnitude --act-bits 2` (no `--use-dplr-loss`)
Init-baseline (FP 90.88%). Ô sạch magnitude/no-DPLR của bộ **br0.95 + act2**. Bỏ qua FIM extraction
(partition=magnitude, không DPLR). **Cao nhất trong 3 run br0.95/act2** — magnitude/no-DPLR (82.63%)
còn vượt cả fim+DPLR (80.53%).

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.95 --apb-scope full --epochs 29 --fim-batches 10 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 2 --partition magnitude
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.26% | 5.43% |
| **Best (ep28)** | **82.63%** | 96.82% |

- quant cost vs FP 90.88% = **−8.25%** | eff-bit W (whole) = 4.29, packed **26.2 MB**, comp **9.3×** (Eq10 APB = 3.61 bit)
- APB: 100 layers, avg binary=0.950, avg eff_bits=3.55 | act LSQ signed 2-bit trên 100 APB-Linear inputs
- ~1228s/ep × 29 ≈ **9.9h**. Best ep28 (82.63), ep29 hơi tụt (82.13) → ~plateau ep25-29 (82.3-82.6).

## ⭐ Cặp SẠCH (chỉ khác partition, cùng no-DPLR / br0.95 / act2)
| run | partition | top-1 |
|-----|-----------|-------|
| fim | fim | 69.75% |
| **này** | **magnitude** | **82.63%** |

→ **magnitude − fim = +12.88%**. So br0.99/act2/no-DPLR (+12.49%): gap **KHÔNG co lại** khi br giảm.
magnitude thắng fim **9/9 ô** đã đo (8 ô br0.99 + ô này). Finding cực robust qua cả br.

## ⭐ magnitude/no-DPLR còn thắng fim+DPLR (cùng br0.95/act2)
| run | partition | DPLR | top-1 |
|-----|-----------|------|-------|
| **này** | magnitude | ✗ | **82.63%** |
| fim+dplr | fim | ✓ | 80.53% |

→ magnitude **không cần distillation** vẫn hơn fim **có** distillation +2.10%. Củng cố: giá trị nằm ở
magnitude-partition, distillation (FIM-loss) chỉ vá cho fim-partition tệ, không đủ đuổi kịp magnitude.

## So br0.99 (cùng ô magnitude/no-DPLR/act2)
| br | top-1 | Δ |
|----|-------|---|
| 0.99 | 77.98% | — |
| **0.95** | **82.63%** | **+4.65%** |

## Per-epoch (đầy đủ) — no-DPLR (CE-only)
```
Ep  1: train_loss=2.6646 | top1=55.17% top5=84.93% | 1228.4s
Ep  2: train_loss=1.4593 | top1=63.43% top5=90.14% | 1227.7s
Ep  3: train_loss=1.1821 | top1=67.72% top5=92.24% | 1227.8s
Ep  4: train_loss=1.0269 | top1=70.53% top5=93.42% | 1229.4s
Ep  5: train_loss=0.9165 | top1=72.94% top5=94.15% | 1230.1s
Ep  6: train_loss=0.8211 | top1=72.85% top5=94.19% | 1229.9s
Ep  7: train_loss=0.7579 | top1=74.41% top5=94.65% | 1228.1s
Ep  8: train_loss=0.6925 | top1=74.99% top5=94.73% | 1227.6s
Ep  9: train_loss=0.6456 | top1=76.11% top5=95.04% | 1227.7s
Ep 10: train_loss=0.5875 | top1=76.24% top5=95.25% | 1228.1s
Ep 11: train_loss=0.5471 | top1=76.36% top5=95.45% | 1229.1s
Ep 12: train_loss=0.4995 | top1=78.05% top5=95.84% | 1228.7s
Ep 13: train_loss=0.4616 | top1=78.22% top5=95.61% | 1228.4s
Ep 14: train_loss=0.4226 | top1=78.45% top5=96.02% | 1228.3s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: train_loss=0.3798 | top1=79.04% top5=96.15% | 1227.3s
Ep 16: train_loss=0.3457 | top1=79.33% top5=96.16% | 1226.5s
Ep 17: train_loss=0.3185 | top1=79.76% top5=96.43% | 1226.5s
Ep 18: train_loss=0.2915 | top1=79.94% top5=96.36% | 1229.5s
Ep 19: train_loss=0.2606 | top1=80.35% top5=96.52% | 1230.3s
Ep 20: train_loss=0.2352 | top1=81.16% top5=96.65% | 1229.5s
Ep 21: train_loss=0.2144 | top1=80.56% top5=96.41% | 1229.7s
Ep 22: train_loss=0.1914 | top1=81.43% top5=96.67% | 1229.7s
Ep 23: train_loss=0.1777 | top1=81.80% top5=96.74% | 1227.3s
Ep 24: train_loss=0.1644 | top1=82.00% top5=96.67% | 1225.4s
Ep 25: train_loss=0.1549 | top1=82.27% top5=96.99% | 1226.4s
Ep 26: train_loss=0.1436 | top1=82.58% top5=97.00% | 1226.5s
Ep 27: train_loss=0.1341 | top1=82.29% top5=96.99% | 1229.9s
Ep 28: train_loss=0.1308 | top1=82.63% top5=96.82% | 1227.4s   ← BEST
Ep 29: train_loss=0.1242 | top1=82.13% top5=96.96% | 1229.9s
```
