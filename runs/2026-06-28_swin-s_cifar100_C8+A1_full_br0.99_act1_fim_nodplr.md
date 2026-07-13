# Swin-S APB W+A — CIFAR-100 (C8+A1 **FIM / no-DPLR** / full / br0.99 / act 1-bit)

> ⛔ **DEPRECATED partition (2026-07-13):** `--partition fim` ở đây = **dplr** (bản tự chế) → số partition KHÔNG dùng làm bằng chứng nữa; đại diện FIM đúng = `fisher` (`--partition fisher`) — A2 đã có (Kaggle): **magnitude > fisher** (xem `2026-07-03_..._A2_fisher_partition...` + `../ABLATIONS.md`).

Run date: 2026-06-28 (~7.5h) | Script: `apb_fimaq/qat.py` | `--partition fim`, KHÔNG `--use-dplr-loss`
Cell cuối của ma trận 2×2 (partition × DPLR). FIM chỉ dùng cho **partition**, không có distillation loss.

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full --epochs 29 --fim-batches 10 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 1 --partition fim
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.17% | 5.51% |
| **Best (ep29)** | **45.10%** | 75.88% |

- quant cost vs FP 90.88% = **−45.78%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~928s/epoch × 29 ≈ **7.5h**. val vẫn nhích nhẹ ở ep29 (44.4→45.1) — gần plateau.

## ⭐⭐ MA TRẬN 2×2 HOÀN CHỈNH (partition × DPLR) — act1 / full / br0.99 / 29ep / seed3407
| partition \ DPLR | **no-DPLR** | **DPLR** | → DPLR giúp |
|------------------|-------------|----------|-------------|
| **fim** | **45.10%** | 51.03% | +5.93% |
| **magnitude** | 48.87% | **58.29%** | +9.42% |
| **→ magnitude − fim** | **+3.77%** | **+7.26%** | |

### Kết luận chính (ở chế độ cực hạn act1/br0.99/full)
1. **magnitude > fim ở CẢ HAI cột** (+3.77% no-DPLR, +7.26% DPLR) → finding magnitude>FIM **robust**, không
   phải nhờ DPLR. ĐI NGƯỢC giả thuyết "FIM partition tốt hơn".
2. **FIM-cho-partition đơn thuần TỆ HƠN magnitude** (45.10 < 48.87, cùng no-DPLR, sạch tuyệt đối) → ở mức
   binarize cực hạn, dùng FIM chọn outlier **hại** so với chỉ giữ |w| lớn.
3. **FIM vẫn có giá trị — nhưng ở DISTILLATION, không phải partition.** DPLR (loss dùng FIM) giúp cả hai
   (+5.93 fim, +9.42 magnitude). Tức **tách vai trò FIM**: kém khi chọn partition, tốt khi làm tín hiệu distill.
4. **Tương tác super-additive:** khoảng cách magnitude−fim NỚI RỘNG khi bật DPLR (3.77 → 7.26). magnitude +
   DPLR cộng hưởng → cấu hình tốt nhất 58.29%.

### Giả thuyết vì sao FIM-partition thua magnitude
APB binarize α·sign(w) → **sai số tái tạo lớn nhất ở weight |lớn|** (xa ±α nhất). magnitude giữ đúng nhóm đó
làm FP → giảm trực tiếp sai số; ở br0.99 (chỉ 1% được FP) + act1, sai số tái tạo **lấn át** lợi ích curvature
của FIM. FIM chọn high-curvature ≠ |lớn| → để lọt weight lớn vào vùng binary → hỏng nặng hơn.

### ⚠️ Cảnh báo cho thesis
- Đây là **config cực hạn NHẤT** (act1 + br0.99 + full). Rất có thể FIM **lật lại** magnitude ở config nhẹ hơn
  (br0.95, act≥2) — khi sai số tái tạo không còn áp đảo. **PHẢI chạy** br0.95/act2 trước khi kết luận.
- single seed (3407) — xác nhận thêm seed cho finding magnitude>FIM.
- Nếu FIM thua magnitude ở MỌI config → đóng góp "FIM-guided partition" cần định vị lại (vd chuyển trọng tâm
  sang FIM-distillation, vốn vẫn thắng rõ).

## Per-epoch (đầy đủ) — CE-only
```
Ep  1: train_loss=4.1996 | top1=10.47% top5=31.78% | 931.8s
Ep  2: train_loss=3.7022 | top1=14.27% top5=39.57% | 931.7s
Ep  3: train_loss=3.4570 | top1=18.46% top5=45.15% | 930.5s
Ep  4: train_loss=3.2518 | top1=21.04% top5=48.28% | 927.9s
Ep  5: train_loss=3.0959 | top1=23.07% top5=52.38% | 927.5s
Ep  6: train_loss=2.9528 | top1=25.27% top5=55.56% | 928.7s
Ep  7: train_loss=2.8408 | top1=28.38% top5=58.56% | 929.5s
Ep  8: train_loss=2.7380 | top1=29.89% top5=60.74% | 927.8s
Ep  9: train_loss=2.6427 | top1=31.71% top5=62.41% | 930.4s
Ep 10: train_loss=2.5434 | top1=32.78% top5=63.91% | 929.2s
Ep 11: train_loss=2.4598 | top1=34.24% top5=65.42% | 929.1s
Ep 12: train_loss=2.3844 | top1=35.66% top5=66.95% | 928.6s
Ep 13: train_loss=2.3134 | top1=36.95% top5=68.76% | 928.7s
Ep 14: train_loss=2.2545 | top1=38.06% top5=69.15% | 929.3s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: train_loss=2.1864 | top1=38.40% top5=69.94% | 928.1s
Ep 16: train_loss=2.1266 | top1=39.81% top5=71.29% | 927.4s
Ep 17: train_loss=2.0759 | top1=40.17% top5=71.85% | 927.6s
Ep 18: train_loss=2.0283 | top1=40.66% top5=71.13% | 927.9s
Ep 19: train_loss=1.9784 | top1=42.02% top5=73.15% | 928.1s
Ep 20: train_loss=1.9269 | top1=41.90% top5=73.46% | 928.0s
Ep 21: train_loss=1.8892 | top1=42.64% top5=73.90% | 927.8s
Ep 22: train_loss=1.8468 | top1=43.29% top5=74.55% | 927.1s
Ep 23: train_loss=1.8122 | top1=43.45% top5=75.02% | 926.6s
Ep 24: train_loss=1.7765 | top1=44.49% top5=75.23% | 928.1s
Ep 25: train_loss=1.7561 | top1=44.81% top5=75.70% | 928.1s
Ep 26: train_loss=1.7291 | top1=44.38% top5=75.41% | 927.3s
Ep 27: train_loss=1.7125 | top1=44.57% top5=75.38% | 927.9s
Ep 28: train_loss=1.6881 | top1=45.01% top5=75.76% | 928.4s
Ep 29: train_loss=1.6845 | top1=45.10% top5=75.88% | 927.1s   ← BEST
```

## TODO (ưu tiên cao → thấp)
- **br0.95 + act2** chạy lại 2×2 (hoặc ít nhất fim vs magnitude, DPLR) — kiểm FIM có lật lại không khi bớt cực hạn.
- magnitude+DPLR đã verify reproducible (chạy 2 lần đều 58.29%).
- seed khác cho finding magnitude>FIM.
