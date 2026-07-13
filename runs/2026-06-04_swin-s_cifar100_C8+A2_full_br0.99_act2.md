# Swin-S APB W+A quant — CIFAR-100 (C8+A2: full / br0.99 / DPLR / **act 2-bit**)

> ⛔ **DEPRECATED partition (2026-07-13):** partition mặc định lúc này = **fim = dplr** (bản tự chế) → số partition KHÔNG dùng làm bằng chứng nữa; đại diện FIM đúng = `fisher` (`--partition fisher`) — A2 đã có (Kaggle): **magnitude > fisher** (xem `2026-07-03_..._A2_fisher_partition...` + `../ABLATIONS.md`). *(DPLR **loss** vẫn hợp lệ.)*

Run date: 2026-06-04 → 2026-06-05 (~11.7h) | Script: `apb_fimaq/qat.py`
Activation quant LSQ **2-bit** (scope A = input 100 APB Linear, full). Base = C8 (weight-only).

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full \
  --use-dplr-loss --dplr-lambda 3000 --epochs 29 --fim-batches 10 \
  --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 2
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.09% | 4.96% |
| **Best (W+A2, ep29)** | **78.49%** | 95.74% |

- quant cost vs FP 90.88% = **−12.39%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- ~1450s/epoch × 29 ≈ 11.7h. 1562 it/ep (batch 32).

## So với các act-bit khác (full br0.99 DPLR, batch 32 — ablation sạch)
| run | A bit | Best top-1 | Δ vs A8 |
|-----|-------|-----------|---------|
| C8+A8 | 8 | 88.94% | — |
| **C8+A2** | **2** | **78.49%** | **−10.45%** |

→ Hạ activation 8→2 bit mất **−10.45%**. 2-bit activation trên Swin rất khắc nghiệt (outlier ViT).
**val CHƯA plateau ở ep29** (78.11→78.49) → chạy thêm epoch có thể gỡ thêm chút.

## Per-epoch (đầy đủ)
```
Ep  1: loss=4.6327 (ce=3.7549) | top1=23.61% top5=53.08% | 1448.3s
Ep  2: loss=3.4722 (ce=2.6237) | top1=41.02% top5=74.41% | 1451.5s
Ep  3: loss=2.8227 (ce=1.9920) | top1=52.83% top5=82.33% | 1451.5s
Ep  4: loss=2.4680 (ce=1.6505) | top1=57.36% top5=86.41% | 1452.2s
Ep  5: loss=2.2374 (ce=1.4301) | top1=59.55% top5=88.24% | 1452.7s
Ep  6: loss=2.0865 (ce=1.2867) | top1=64.60% top5=90.40% | 1452.2s
Ep  7: loss=1.9649 (ce=1.1731) | top1=65.00% top5=90.53% | 1450.9s
Ep  8: loss=1.8507 (ce=1.0658) | top1=66.84% top5=91.18% | 1451.1s
Ep  9: loss=1.7538 (ce=0.9766) | top1=67.66% top5=92.06% | 1451.9s
Ep 10: loss=1.6664 (ce=0.8952) | top1=69.01% top5=92.31% | 1454.6s
Ep 11: loss=1.5977 (ce=0.8332) | top1=70.58% top5=93.28% | 1452.5s
Ep 12: loss=1.5362 (ce=0.7778) | top1=70.40% top5=93.18% | 1451.5s
Ep 13: loss=1.4647 (ce=0.7121) | top1=71.31% top5=92.79% | 1451.9s
Ep 14: loss=1.3991 (ce=0.6525) | top1=72.16% top5=93.33% | 1452.2s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: loss=1.3382 (ce=0.5971) | top1=73.09% top5=94.21% | 1450.0s
Ep 16: loss=1.2832 (ce=0.5464) | top1=73.94% top5=94.40% | 1450.1s
Ep 17: loss=1.2355 (ce=0.5026) | top1=74.58% top5=94.44% | 1449.7s
Ep 18: loss=1.1905 (ce=0.4609) | top1=74.75% top5=94.73% | 1449.7s
Ep 19: loss=1.1463 (ce=0.4199) | top1=75.40% top5=94.83% | 1450.9s
Ep 20: loss=1.1064 (ce=0.3829) | top1=75.79% top5=94.90% | 1450.9s
Ep 21: loss=1.0705 (ce=0.3498) | top1=76.38% top5=95.13% | 1451.1s
Ep 22: loss=1.0369 (ce=0.3181) | top1=76.54% top5=95.43% | 1450.4s
Ep 23: loss=1.0125 (ce=0.2963) | top1=77.23% top5=95.40% | 1450.5s
Ep 24: loss=0.9889 (ce=0.2747) | top1=77.37% top5=95.74% | 1449.9s
Ep 25: loss=0.9619 (ce=0.2491) | top1=77.18% top5=95.81% | 1449.3s
Ep 26: loss=0.9412 (ce=0.2296) | top1=77.49% top5=95.29% | 1450.7s
Ep 27: loss=0.9296 (ce=0.2186) | top1=78.13% top5=95.70% | 1450.5s
Ep 28: loss=0.9220 (ce=0.2114) | top1=78.11% top5=95.61% | 1451.9s
Ep 29: loss=0.9115 (ce=0.2012) | top1=78.49% top5=95.74% | 1453.9s   ← BEST (vẫn tăng)
```

## TODO
- **C8+A4** để lấp khoảng A8 (88.94%) ↔ A2 (78.49%).
- C8+A2 chạy thêm epoch (40+) xem gỡ được tới đâu (val chưa plateau).
- Chạy lại C8 weight-only batch 32 để tách confound batch.
- Lặp cho scope skip (C6+A2) — config accuracy cao nhất.
