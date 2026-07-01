# Swin-S APB W+A quant — CIFAR-100 (C8+A1: full / br0.99 / DPLR / **act 1-bit binary**)

Run date: 2026-06-28 (~9.3h) | Script: `apb_fimaq/qat.py`
Activation quant **BinaryActQuant 1-bit** (sign·scale, XNOR/Bi-Real STE; scope A = input 100 APB Linear, full).
Base = C8 (weight-only). **Khác A2/A8: 1-bit không dùng LSQ** (LSQ degenerate ở 1-bit, Qp=0) → routing sang
`BinaryActQuant` (out = scale·sign(x), scale init = mean|x|). Xem commit `8f9b718`.

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full \
  --use-dplr-loss --dplr-lambda 3000 --epochs 29 --fim-batches 10 \
  --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 1
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.17% | 5.51% |
| **Best (W+A1, ep28-29)** | **51.03%** | 80.84% |

- quant cost vs FP 90.88% = **−39.85%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~1154s/epoch × 29 ≈ **9.3h**. 1562 it/ep (batch 32), 1.47 it/s.

## So với các act-bit khác (full br0.99 DPLR, batch 32 — ablation sạch)
| run | A bit | quantizer | Best top-1 | Δ vs A8 | Δ bước trước |
|-----|-------|-----------|-----------|---------|--------------|
| C8+A8 | 8 | LSQ | 88.94% | — | — |
| C8+A2 | 2 | LSQ | 78.49% | −10.45% | A8→A2 −10.45% |
| **C8+A1** | **1** | **Binary** | **51.03%** | **−37.91%** | **A2→A1 −27.46%** |

→ Vách đá ở 1-bit: A2→A1 mất **−27.46%**, dốc gấp ~2.6× so với A8→A2 (−10.45%). Đúng bản chất binary
activation: vứt toàn bộ magnitude, chỉ giữ dấu → ViT (activation 2 phía sau LN/GELU, nhiều outlier) chịu
cú sốc mạnh nhất. 51% vẫn >> random (1%) → QAT + DPLR distillation kéo từ 1.17% lên 51%, có sự sống.

**val ĐÃ plateau ~ep23-29** (50.07→50.61→50.18→50.37→50.71→51.03→51.03) → khác A2 (chưa plateau). 1-bit
gần như hết cải thiện ở ep29, chạy thêm epoch lợi rất ít. Lần sau ~22-24 ep là đủ.

## ⚠️ Tốc độ: 1-bit NHANH hơn 2-bit (ngược trực giác)
| run | s/epoch | tổng |
|-----|---------|------|
| C8+A2 (LSQ) | ~1450s | ~11.7h |
| **C8+A1 (Binary)** | **~1154s** | **~9.3h** |

1-bit nhanh hơn ~20%/epoch. Lý do: LSQ forward nhiều op hơn (div+clamp+round+mul+grad_scale, mỗi cái 1
kernel + node autograd) còn Binary chỉ 1 custom Function (sign·scale). Tốc độ 1-bit **ổn định tuyệt đối**
(1152-1158s suốt 29 ep) → KHÔNG có memory thrashing. (Cảm giác "1-bit siêu lâu" trước đó là do so với run
khác config / quan sát giữa chừng, không phải dữ liệu — thực đo thì 1-bit nhanh hơn.)

## Lưu ý log: 3 số loss không cộng khớp
`loss=4.5519` nhưng `ce=4.1489 + λ·dplr=0.0001` → lệch ~0.4 (đều ở mọi epoch, ep29: 1.8685 vs 1.4854).
Nhiều khả năng số in `dplr=0.0001` là **dplr thô chưa nhân λ**; đóng góp thật λ·dplr ≈ 0.38-0.40 (≈ phần
lệch). Distillation CÓ tác dụng nhưng nhãn hiển thị sai → nên sửa format log cho cộng đúng.

## Per-epoch (đầy đủ)
```
Ep  1: loss=4.5519 (ce=4.1489) | top1=10.82% top5=32.48% | 1152.1s
Ep  2: loss=4.0506 (ce=3.6521) | top1=16.35% top5=41.20% | 1153.8s
Ep  3: loss=3.7770 (ce=3.3790) | top1=20.06% top5=47.06% | 1153.3s
Ep  4: loss=3.5752 (ce=3.1781) | top1=23.10% top5=52.19% | 1153.7s
Ep  5: loss=3.3990 (ce=3.0031) | top1=25.99% top5=55.70% | 1155.0s
Ep  6: loss=3.2409 (ce=2.8460) | top1=30.00% top5=60.13% | 1156.0s
Ep  7: loss=3.1097 (ce=2.7160) | top1=32.66% top5=64.09% | 1155.7s
Ep  8: loss=2.9902 (ce=2.5975) | top1=33.10% top5=63.87% | 1155.8s
Ep  9: loss=2.8781 (ce=2.4864) | top1=35.86% top5=68.26% | 1156.0s
Ep 10: loss=2.7731 (ce=2.3823) | top1=36.66% top5=69.11% | 1155.8s
Ep 11: loss=2.6901 (ce=2.3001) | top1=39.68% top5=70.98% | 1155.5s
Ep 12: loss=2.6085 (ce=2.2193) | top1=40.46% top5=71.62% | 1154.7s
Ep 13: loss=2.5293 (ce=2.1413) | top1=39.89% top5=71.76% | 1155.4s
Ep 14: loss=2.4638 (ce=2.0763) | top1=42.36% top5=73.68% | 1155.3s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: loss=2.3892 (ce=2.0026) | top1=43.46% top5=75.48% | 1154.0s
Ep 16: loss=2.3224 (ce=1.9363) | top1=45.57% top5=76.35% | 1154.3s
Ep 17: loss=2.2697 (ce=1.8842) | top1=46.87% top5=77.03% | 1153.5s
Ep 18: loss=2.2207 (ce=1.8357) | top1=46.56% top5=77.36% | 1154.5s
Ep 19: loss=2.1685 (ce=1.7840) | top1=47.43% top5=78.04% | 1154.0s
Ep 20: loss=2.1254 (ce=1.7411) | top1=48.02% top5=78.23% | 1154.7s
Ep 21: loss=2.0835 (ce=1.6995) | top1=48.60% top5=78.81% | 1153.7s
Ep 22: loss=2.0419 (ce=1.6581) | top1=49.47% top5=79.15% | 1154.0s
Ep 23: loss=2.0081 (ce=1.6246) | top1=50.07% top5=79.67% | 1154.0s
Ep 24: loss=1.9743 (ce=1.5910) | top1=50.61% top5=79.91% | 1154.5s
Ep 25: loss=1.9432 (ce=1.5601) | top1=50.18% top5=80.15% | 1156.7s
Ep 26: loss=1.9157 (ce=1.5327) | top1=50.37% top5=80.00% | 1157.7s
Ep 27: loss=1.9008 (ce=1.5178) | top1=50.71% top5=80.42% | 1157.6s
Ep 28: loss=1.8818 (ce=1.4989) | top1=51.03% top5=80.78% | 1156.9s   ← BEST top1 (đạt lần đầu)
Ep 29: loss=1.8685 (ce=1.4854) | top1=51.03% top5=80.84% | 1154.5s   (plateau)
```

## TODO
- **C8+A4** vẫn cần (lấp A8 88.94% ↔ A2 78.49%); giờ có thêm điểm neo A1=51.03% → đường A8/A4/A2/A1.
- Sửa nhãn log `λ·dplr` cho 3 số cộng khớp (đang in dplr thô).
- 1-bit nếu muốn cứu thêm: thử scope **skip** (bỏ head.fc khỏi act-quant) hoặc binary_ratio thấp hơn —
  nhưng 1-bit act trên ViT có thể là trần cứng ~50%.
- ⚠️ single seed (3407).
