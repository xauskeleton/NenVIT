# Swin-S APB W+A — CIFAR-100 (C8+A1 **MAGNITUDE partition** / full / br0.99 / DPLR / act 1-bit)

Run date: 2026-06-28 (~9.1h) | Script: `apb_fimaq/qat.py` | flag mới `--partition magnitude`
**Ablation partition:** giống hệt C8+A1 (fim) nhưng phân vùng FP-outlier theo **|w|** (APB gốc) thay vì FIMA-Q.

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full \
  --use-dplr-loss --dplr-lambda 3000 --epochs 29 --fim-batches 10 \
  --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 1 --partition magnitude
```
(FIM vẫn tính vì `--use-dplr-loss` cần cho teacher loss; nhưng partition KHÔNG dùng FIM.)

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 0.94% | 4.94% |
| **Best (ep28)** | **58.29%** | 85.58% |

- quant cost vs FP 90.88% = **−32.59%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~1128s/epoch × 29 ≈ **9.1h**. val plateau ~ep26-29 (57.9→58.3).

## ⭐ FINDING: magnitude > FIM ở config cực hạn này
| partition | DPLR | Best top-1 | Δ |
|-----------|------|-----------|---|
| **fim** | ✓ | 51.03% | — |
| **magnitude** | ✓ | **58.29%** | **+7.26%** |

→ Cùng mọi thứ, chỉ khác tiêu chí phân vùng → **magnitude hơn FIM +7.26%**. ĐI NGƯỢC giả thuyết
"FIMA-Q partition tốt hơn". Đây là 1 cell trong ablation 2×2 (partition × dplr).

**Giả thuyết vì sao** (cần kiểm chứng thêm): APB binarize bằng α·sign(w) → **sai số nhị phân hóa lớn nhất
ở weight |lớn|** (chúng xa ±α nhất). Giữ weight |lớn| làm FP (magnitude) → giảm trực tiếp sai số tái tạo;
ở br0.99 + act 1-bit (chế độ cực hạn) sai số tái tạo có thể **lấn át** thông tin curvature của FIM. FIM chọn
weight high-curvature, không nhất thiết |lớn| → sai số binarize lớn hơn → tệ hơn. (Cũng có thể FIM nhiễu ở
10 batch.) ⚠️ **1 điểm dữ liệu, ở config cực hạn** — KHÔNG kết luận FIM vô dụng; cần thử config nhẹ hơn
(br0.95, act ≥2) xem FIM có lật lại không.

## Per-epoch (đầy đủ)
```
Ep  1: loss=4.6440 (ce=4.1088) | top1=12.36% top5=35.12% | 1126.2s
Ep  2: loss=4.0046 (ce=3.4987) | top1=19.55% top5=47.84% | 1126.8s
Ep  3: loss=3.6496 (ce=3.1506) | top1=24.16% top5=54.55% | 1126.8s
Ep  4: loss=3.4101 (ce=2.9170) | top1=29.62% top5=61.29% | 1127.3s
Ep  5: loss=3.2181 (ce=2.7289) | top1=31.57% top5=62.94% | 1128.5s
Ep  6: loss=3.0589 (ce=2.5734) | top1=35.65% top5=66.89% | 1128.5s
Ep  7: loss=2.8984 (ce=2.4150) | top1=38.00% top5=69.73% | 1128.6s
Ep  8: loss=2.7711 (ce=2.2904) | top1=39.94% top5=71.41% | 1128.4s
Ep  9: loss=2.6430 (ce=2.1644) | top1=42.34% top5=74.60% | 1128.2s
Ep 10: loss=2.5483 (ce=2.0718) | top1=44.00% top5=75.83% | 1128.3s
Ep 11: loss=2.4553 (ce=1.9808) | top1=46.33% top5=77.10% | 1128.8s
Ep 12: loss=2.3739 (ce=1.9013) | top1=48.19% top5=78.35% | 1128.6s
Ep 13: loss=2.3030 (ce=1.8320) | top1=46.34% top5=77.68% | 1128.9s
Ep 14: loss=2.2303 (ce=1.7609) | top1=49.15% top5=79.44% | 1128.7s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: loss=2.1695 (ce=1.7011) | top1=51.29% top5=80.74% | 1127.8s
Ep 16: loss=2.1050 (ce=1.6375) | top1=51.64% top5=81.47% | 1127.5s
Ep 17: loss=2.0497 (ce=1.5831) | top1=52.08% top5=82.07% | 1127.6s
Ep 18: loss=1.9937 (ce=1.5277) | top1=52.99% top5=82.36% | 1127.7s
Ep 19: loss=1.9499 (ce=1.4851) | top1=54.43% top5=83.39% | 1127.5s
Ep 20: loss=1.9086 (ce=1.4441) | top1=54.73% top5=83.85% | 1128.1s
Ep 21: loss=1.8600 (ce=1.3959) | top1=55.37% top5=83.97% | 1128.0s
Ep 22: loss=1.8224 (ce=1.3582) | top1=55.98% top5=84.49% | 1128.3s
Ep 23: loss=1.7874 (ce=1.3235) | top1=56.70% top5=85.06% | 1128.0s
Ep 24: loss=1.7540 (ce=1.2901) | top1=56.61% top5=85.29% | 1127.3s
Ep 25: loss=1.7315 (ce=1.2679) | top1=57.18% top5=85.19% | 1127.2s
Ep 26: loss=1.7027 (ce=1.2389) | top1=57.91% top5=85.14% | 1128.0s
Ep 27: loss=1.6829 (ce=1.2191) | top1=57.93% top5=85.71% | 1128.4s
Ep 28: loss=1.6724 (ce=1.2084) | top1=58.29% top5=85.58% | 1128.4s   ← BEST
Ep 29: loss=1.6603 (ce=1.1962) | top1=58.22% top5=85.63% | 1127.4s
```

## TODO (hoàn tất ablation 2×2 partition × dplr, act-1)
- [x] fim + dplr = **51.03%** (`..._C8+A1_full_br0.99_act1.md`)
- [x] magnitude + dplr = **58.29%** (file này)
- [ ] **fim + no-dplr** (Test 3): `--partition fim` bỏ `--use-dplr-loss`
- [ ] **magnitude + no-dplr** (Test 2): `--partition magnitude` bỏ `--use-dplr-loss`
- Sau đó: kiểm config nhẹ hơn (br0.95 / act≥2) xem FIM có thắng lại magnitude không.
- ⚠️ single seed (3407); finding magnitude>FIM cần xác nhận seed khác.
