# Swin-S APB W+A quant — CIFAR-100 (C8+A8: full / br0.99 / DPLR / **act 8-bit**)

Run date: 2026-06-03 → kết thúc 2026-06-04 (~12h) | Script: `apb_fimaq/qat.py`
**Run đầu tiên có activation quant** (LSQ 8-bit, scope A = input các APB Linear). Base = C8 (weight-only).

## Command (ước lượng từ config + log)
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full \
  --use-dplr-loss --dplr-lambda 3000 --epochs 30 --fim-batches 10 \
  --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 8
```
⚠️ `--batch-size 32` (không phải 64) vì batch 64 + act-quant + DPLR teacher bị **OOM** trên Kaggle GPU.

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| **Best (W+A8)** | **88.94%** | — |
| ep30 final | 88.86% | 98.45% |

- **quant cost (vs FP 90.88%) = −1.94%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~1424s/epoch (≈23.7 phút) × 30 = ~12h. Chậm gần 2× so C8 do batch32 + act-quant + teacher.

## So với C8 (weight-only, full br0.99 DPLR, batch64) = 88.55%
**+0.39%** dù thêm quantize activation → act 8-bit gần như free ở mức bit này.

⚠️ **CONFOUND:** C8+A8 khác C8 ở **2** chỗ: (1) thêm act-quant, (2) batch 32 vs 64. Batch nhỏ = gấp đôi
gradient step/epoch, tự nó cũng tăng acc → KHÔNG quy hết +0.39% cho act-quant. Muốn tách phải chạy lại
C8 ở batch 32.

## Per-epoch (chỉ có 2 mốc từ log paste)
```
Ep  2/30: loss=1.4236 (ce=0.7705 + λ·dplr=0.0002) | val top1=77.66% top5=95.93% | 1425.8s
...
Ep 30/30: loss=0.3572 (ce=0.0102 + λ·dplr=0.0001) | val top1=88.86% top5=98.45% | 1423.7s
DONE. Best val top1 = 88.94%
```
(các epoch 1, 3–29 chưa lưu — nếu cần lấy đầy đủ từ `checkpoints/qat/train.log` trên Kaggle.)

## TODO
- Chạy **C8+A4** (act 4-bit) để vẽ đường act-bit ↔ accuracy.
- Chạy lại **C8 batch 32** (weight-only) để tách ảnh hưởng batch khỏi act-quant.
- Giải OOM (custom LSQ autograd.Function lean hơn / grad-checkpoint) để A-quant chạy được batch 64.
- Lặp cho scope **skip** (C6+A8) — config accuracy cao nhất.
