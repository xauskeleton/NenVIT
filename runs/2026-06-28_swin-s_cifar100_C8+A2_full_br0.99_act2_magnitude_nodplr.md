# Swin-S APB W+A — CIFAR-100 (C8+A2 **MAGNITUDE / no-DPLR** / full / br0.99 / act 2-bit)

Run date: 2026-06-28 (~9.7h) | Script: `apb_fimaq/qat.py` | `--partition magnitude --act-bits 2`, KHÔNG `--use-dplr-loss`
Bước kiểm "magnitude có thắng fim khi bớt cực hạn (act2)". ⚠️ Run này **no-DPLR** (LSQ 2-bit, CE-only).

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full --epochs 29 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 2 --partition magnitude
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.12% | 5.37% |
| **Best (ep29)** | **77.98%** | 95.62% |

- quant cost vs FP 90.88% = **−12.90%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~1202s/epoch × 29 ≈ **9.7h**. val gần plateau ep24-29 (77.1→78.0); train_loss rất thấp (0.19) → act2 dễ hơn act1 nhiều, hơi overfit cuối.

## ⚠️ CHƯA phải so sánh sạch với C8+A2 (fim)
| run | act | partition | DPLR | top-1 |
|-----|-----|-----------|------|-------|
| C8+A2 (cũ) | 2 | **fim** | **✓** | 78.49% |
| **này** | 2 | **magnitude** | **✗** | **77.98%** |

→ Hai run khác **CẢ partition LẪN DPLR** → KHÔNG kết luận magnitude vs fim ở act2 từ cặp này. Nhưng đáng chú ý:
**magnitude no-DPLR (77.98) gần bằng fim+DPLR (78.49), chỉ −0.51%** — dù không có distillation. Gợi ý ở act2
magnitude vẫn rất cạnh tranh và DPLR ít quan trọng hơn so với act1.

## So sánh SẠCH với dữ liệu sẵn có
**magnitude / no-DPLR, theo act-bit (cùng partition + cùng no-DPLR):**
| act | top-1 | |
|-----|-------|---|
| 1 | 48.87% | `..._act1_magnitude_nodplr.md` |
| **2** | **77.98%** | file này |

→ act1→act2 (magnitude, no-DPLR) = **+29.11%**. Vách đá act-bit cực lớn, khớp đường A8/A2/A1 trước đó.

**magnitude > fim ở act1/no-DPLR** đã có (48.87 > 45.10, +3.77%). Để biết có giữ ở act2 cần **act2/fim/no-DPLR**.

## ✅ So sánh act2 ĐÃ KHÉP (cả 2 cặp sạch đã chạy)
- ✅ **act2 / fim / no-DPLR** = 65.49% → cặp sạch với run này (77.98) ⇒ magnitude > fim **+12.49%** (no-DPLR).
- ✅ **act2 / magnitude / DPLR** = **81.79%** (`2026-06-30_..._act2_magnitude.md`) → cặp sạch với C8+A2 fim+DPLR
  (78.49) ⇒ magnitude > fim **+3.30%** (DPLR). 2×2 partition×DPLR ở act2 nay đầy đủ (xem README).

## ⚡ Tốc độ: act2 (LSQ) chậm hơn act1 (Binary)
~1202s/ep (LSQ 2-bit, no-DPLR) vs act1 magnitude no-DPLR ~928s/ep → LSQ nặng hơn Binary ~30%. Thậm chí act2
no-DPLR (~1202s) còn ≈ act1 **có** DPLR (~1154s). Khớp finding "LSQ nhiều op hơn Binary".

## Per-epoch (đầy đủ) — CE-only
```
Ep  1: train_loss=3.7331 | top1=22.21% top5=52.52% | 1200.9s
Ep  2: train_loss=2.6626 | top1=39.76% top5=72.51% | 1204.1s
Ep  3: train_loss=1.9910 | top1=50.40% top5=82.16% | 1203.9s
Ep  4: train_loss=1.6190 | top1=57.11% top5=86.66% | 1202.0s
Ep  5: train_loss=1.3889 | top1=61.64% top5=88.61% | 1199.9s
Ep  6: train_loss=1.2388 | top1=63.42% top5=89.71% | 1199.2s
Ep  7: train_loss=1.1181 | top1=65.71% top5=90.72% | 1200.8s
Ep  8: train_loss=1.0144 | top1=66.42% top5=91.59% | 1203.1s
Ep  9: train_loss=0.9288 | top1=69.29% top5=92.24% | 1200.6s
Ep 10: train_loss=0.8621 | top1=69.11% top5=92.49% | 1201.1s
Ep 11: train_loss=0.7837 | top1=70.60% top5=93.26% | 1200.6s
Ep 12: train_loss=0.7231 | top1=71.91% top5=93.74% | 1201.5s
Ep 13: train_loss=0.6590 | top1=71.86% top5=94.10% | 1201.1s
Ep 14: train_loss=0.6082 | top1=72.73% top5=93.96% | 1201.4s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: train_loss=0.5474 | top1=73.21% top5=93.82% | 1199.6s
Ep 16: train_loss=0.4966 | top1=74.40% top5=94.64% | 1200.8s
Ep 17: train_loss=0.4618 | top1=74.94% top5=94.37% | 1202.9s
Ep 18: train_loss=0.4164 | top1=75.42% top5=94.86% | 1201.9s
Ep 19: train_loss=0.3834 | top1=75.10% top5=94.64% | 1202.3s
Ep 20: train_loss=0.3474 | top1=75.87% top5=94.96% | 1201.6s
Ep 21: train_loss=0.3150 | top1=76.33% top5=95.06% | 1204.0s
Ep 22: train_loss=0.2890 | top1=76.43% top5=95.22% | 1203.1s
Ep 23: train_loss=0.2644 | top1=76.84% top5=95.30% | 1202.1s
Ep 24: train_loss=0.2434 | top1=77.58% top5=95.38% | 1201.6s
Ep 25: train_loss=0.2225 | top1=77.12% top5=95.61% | 1202.5s
Ep 26: train_loss=0.2069 | top1=77.42% top5=95.52% | 1203.2s
Ep 27: train_loss=0.2006 | top1=77.61% top5=95.71% | 1207.9s
Ep 28: train_loss=0.1930 | top1=77.56% top5=95.66% | 1208.9s
Ep 29: train_loss=0.1901 | top1=77.98% top5=95.62% | 1206.6s   ← BEST
```
