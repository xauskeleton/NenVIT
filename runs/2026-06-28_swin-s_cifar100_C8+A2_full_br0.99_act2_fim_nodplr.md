# Swin-S APB W+A — CIFAR-100 (C8+A2 **FIM / no-DPLR** / full / br0.99 / act 2-bit)

Run date: 2026-06-28 (~10h) | Script: `apb_fimaq/qat.py` | `--partition fim --act-bits 2`, KHÔNG `--use-dplr-loss`
Cặp SẠCH với act2/magnitude/no-DPLR (77.98) → trả lời "magnitude > fim có giữ ở act2?". FIM chỉ dùng partition.

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full --epochs 29 --fim-batches 10 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 2 --partition fim
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.09% | 4.96% |
| **Best (ep28)** | **65.49%** | 89.80% |

- quant cost vs FP 90.88% = **−25.39%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~1240s/epoch × 29 ≈ **10h**. val plateau ~ep25-29 (64.7→65.5).

## ⭐⭐⭐ FINDING MẠNH: magnitude > fim ở CẢ act1 LẪN act2 (no-DPLR, cặp SẠCH)
| act | fim/no-DPLR | magnitude/no-DPLR | **mag − fim** |
|-----|-------------|-------------------|---------------|
| 1 | 45.10% | 48.87% | **+3.77%** |
| **2** | **65.49%** | **77.98%** | **+12.49%** |

→ **magnitude thắng fim ở mọi act-bit, khoảng cách NỚI RỘNG khi act tăng (3.77 → 12.49).** BÁC BỎ giả thuyết
trước ("fim lật lại khi bớt cực hạn"). Ngược lại: fim-partition càng tệ tương đối ở act2.

### Giải thích vì sao gap nới rộng ở act2
- **act1:** activation 1-bit là nút cổ chai áp đảo → mọi cấu hình đều tệ (45-49%), lựa chọn partition bị che lấp → gap nhỏ.
- **act2:** activation tốt hơn → **sai số tái tạo weight trở thành yếu tố chi phối** → partition lộ rõ ảnh hưởng → gap lớn.
- Cơ chế: APB binarize α·sign(w), sai số ~|w|−α lớn nhất ở weight |lớn|. magnitude giữ đúng nhóm |lớn| làm FP →
  giảm trực tiếp sai số; FIM giữ high-curvature (≠|lớn|) → để lọt weight lớn vào vùng binary → sai số lớn hơn.

### Hệ quả cho thesis — phải đối mặt thẳng thắn
- **FIM-cho-PARTITION KHÔNG giúp** (thua magnitude rõ, nhất quán 2 act-bit). Đây là **kết quả âm** cho phần
  "FIMA-Q-guided partition".
- **FIM-cho-DISTILLATION (DPLR) VẪN giúp** (act1: +5.93 fim / +9.42 mag). → Định vị lại đóng góp: **giá trị của
  FIMA-Q nằm ở distillation loss, KHÔNG ở tiêu chí phân vùng** (ít nhất ở br0.99/full).
- ⚠️ vẫn br0.99/full/single-seed. Nên thử br0.95 (nhiều FP hơn) + seed khác.

## ⚡ Tốc độ
~1240s/ep (LSQ act2, no-DPLR) — chậm hơn act2/magnitude/no-DPLR (~1202s) chút (LSQ + có bước FIM init); chậm
hơn hẳn act1 Binary (~928s). Khớp "LSQ nặng hơn Binary".

## Per-epoch (đầy đủ) — CE-only
```
Ep  1: train_loss=3.9133 | top1=15.43% top5=40.19% | 1236.8s
Ep  2: train_loss=3.2290 | top1=25.57% top5=54.99% | 1237.9s
Ep  3: train_loss=2.7069 | top1=34.09% top5=66.24% | 1237.4s
Ep  4: train_loss=2.3513 | top1=40.73% top5=72.79% | 1237.8s
Ep  5: train_loss=2.0949 | top1=45.56% top5=76.91% | 1238.6s
Ep  6: train_loss=1.8980 | top1=47.86% top5=79.20% | 1239.0s
Ep  7: train_loss=1.7409 | top1=50.73% top5=80.73% | 1239.3s
Ep  8: train_loss=1.6090 | top1=52.01% top5=82.48% | 1239.2s
Ep  9: train_loss=1.4941 | top1=54.27% top5=83.37% | 1239.5s
Ep 10: train_loss=1.3903 | top1=56.45% top5=84.12% | 1238.9s
Ep 11: train_loss=1.2994 | top1=56.37% top5=84.39% | 1239.6s
Ep 12: train_loss=1.2065 | top1=58.24% top5=85.74% | 1240.3s
Ep 13: train_loss=1.1312 | top1=58.53% top5=85.74% | 1240.0s
Ep 14: train_loss=1.0522 | top1=59.50% top5=86.84% | 1240.0s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: train_loss=0.9723 | top1=60.79% top5=87.47% | 1237.9s
Ep 16: train_loss=0.9090 | top1=61.09% top5=87.70% | 1237.9s
Ep 17: train_loss=0.8435 | top1=61.69% top5=88.21% | 1238.0s
Ep 18: train_loss=0.7889 | top1=62.02% top5=88.08% | 1238.5s
Ep 19: train_loss=0.7329 | top1=63.36% top5=88.56% | 1237.9s
Ep 20: train_loss=0.6824 | top1=63.14% top5=88.34% | 1237.7s
Ep 21: train_loss=0.6362 | top1=63.86% top5=88.88% | 1237.9s
Ep 22: train_loss=0.5911 | top1=64.09% top5=89.21% | 1237.5s
Ep 23: train_loss=0.5528 | top1=64.92% top5=89.65% | 1240.4s
Ep 24: train_loss=0.5239 | top1=64.58% top5=89.13% | 1242.9s
Ep 25: train_loss=0.4979 | top1=64.89% top5=89.71% | 1247.3s
Ep 26: train_loss=0.4709 | top1=64.73% top5=89.73% | 1248.0s
Ep 27: train_loss=0.4575 | top1=65.26% top5=89.79% | 1247.2s
Ep 28: train_loss=0.4405 | top1=65.49% top5=89.80% | 1249.0s   ← BEST
Ep 29: train_loss=0.4312 | top1=65.41% top5=89.78% | 1249.3s
```

## TODO
- (tùy) act2/magnitude/DPLR + đã có C8+A2 fim/DPLR (78.49) → 2×2 act2 đầy đủ.
- br0.95 (1% → 5% FP) xem gap magnitude-vs-fim co lại không.
- seed khác.
