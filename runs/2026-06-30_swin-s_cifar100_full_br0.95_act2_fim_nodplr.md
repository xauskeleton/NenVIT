# Swin-S APB W+A — CIFAR-100 (br0.95 / full / act 2-bit / **fim / no-DPLR**)

> ⛔ **DEPRECATED partition (2026-07-13):** `--partition fim` ở đây = **dplr** (bản tự chế) → số partition KHÔNG dùng làm bằng chứng nữa; đại diện FIM đúng = `fisher` (`--partition fisher`) — A2 đã có (Kaggle): **magnitude > fisher** (xem `2026-07-03_..._A2_fisher_partition...` + `../ABLATIONS.md`).

Run date: 2026-06-30 (~9.8h) | Script: `apb_fimaq/qat.py` | `--partition fim --act-bits 2` (no `--use-dplr-loss`)
Init-baseline (FP 90.88%). Đây là 1 trong bộ **br0.95 + act2** (TODO quan trọng README) để kiểm tra
"magnitude > fim có co lại khi bớt cực hạn br0.99→0.95 không". Ô sạch fim/no-DPLR.

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.95 --apb-scope full --epochs 29 --fim-batches 10 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 2 --partition fim
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.16% | 5.20% |
| **Best (ep29)** | **69.75%** | 92.11% |

- quant cost vs FP 90.88% = **−21.13%** | eff-bit W (whole) = 4.29, packed **26.2 MB**, comp **9.3×** (Eq10 APB = 3.61 bit)
- APB: 100 layers, avg binary=0.950, avg eff_bits=3.55 | act LSQ signed 2-bit trên 100 APB-Linear inputs
- ~1213s/ep × 29 ≈ **9.8h**. val vẫn nhích tới ep29 (69.5→69.75, **chưa plateau**); train ce=0.33.

## ⭐ Cặp SẠCH (chỉ khác partition, cùng no-DPLR / br0.95 / act2)
| run | partition | top-1 |
|-----|-----------|-------|
| **này** | **fim** | **69.75%** |
| magnitude | magnitude | 82.63% |

→ **magnitude − fim = +12.88%** ở br0.95/act2/no-DPLR. So br0.99/act2/no-DPLR (+12.49%): gap **KHÔNG co**,
nhích rộng chút. **Bác bỏ** giả thuyết "fim lật lại / gap co khi br 0.99→0.95". magnitude vẫn thắng đậm.

## So br0.99 (cùng ô fim/no-DPLR/act2)
| br | top-1 | Δ |
|----|-------|---|
| 0.99 | 65.49% | — |
| **0.95** | **69.75%** | **+4.26%** |

→ Bớt binarize (0.99→0.95) nâng fim/no-DPLR +4.26% (đúng kỳ vọng: giữ nhiều FP hơn). Nhưng magnitude cũng
nâng tương đương (+4.65%) → gap giữ nguyên.

## Per-epoch (đầy đủ) — no-DPLR (CE-only)
```
Ep  1: train_loss=3.8647 | top1=17.36% top5=43.95% | 1212.5s
Ep  2: train_loss=3.0578 | top1=30.02% top5=61.69% | 1214.0s
Ep  3: train_loss=2.4895 | top1=37.57% top5=70.86% | 1213.8s
Ep  4: train_loss=2.1446 | top1=44.89% top5=76.35% | 1213.1s
Ep  5: train_loss=1.8968 | top1=48.82% top5=80.07% | 1214.0s
Ep  6: train_loss=1.7080 | top1=52.47% top5=82.07% | 1213.6s
Ep  7: train_loss=1.5624 | top1=54.04% top5=83.84% | 1214.5s
Ep  8: train_loss=1.4315 | top1=56.80% top5=84.93% | 1213.8s
Ep  9: train_loss=1.3146 | top1=58.49% top5=85.82% | 1214.0s
Ep 10: train_loss=1.2199 | top1=58.79% top5=86.05% | 1213.2s
Ep 11: train_loss=1.1223 | top1=61.03% top5=87.31% | 1215.0s
Ep 12: train_loss=1.0404 | top1=62.30% top5=88.44% | 1214.1s
Ep 13: train_loss=0.9605 | top1=62.73% top5=88.67% | 1213.6s
Ep 14: train_loss=0.8814 | top1=64.50% top5=89.40% | 1213.7s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: train_loss=0.8099 | top1=63.79% top5=89.44% | 1212.1s
Ep 16: train_loss=0.7524 | top1=65.15% top5=90.00% | 1211.9s
Ep 17: train_loss=0.6841 | top1=65.66% top5=90.39% | 1212.3s
Ep 18: train_loss=0.6361 | top1=65.95% top5=90.41% | 1212.8s
Ep 19: train_loss=0.5818 | top1=66.43% top5=90.89% | 1212.6s
Ep 20: train_loss=0.5365 | top1=66.28% top5=90.93% | 1211.7s
Ep 21: train_loss=0.4976 | top1=67.49% top5=91.15% | 1212.5s
Ep 22: train_loss=0.4647 | top1=67.82% top5=91.36% | 1211.9s
Ep 23: train_loss=0.4284 | top1=68.09% top5=91.61% | 1213.0s
Ep 24: train_loss=0.4039 | top1=68.66% top5=91.63% | 1212.2s
Ep 25: train_loss=0.3791 | top1=68.93% top5=91.64% | 1211.9s
Ep 26: train_loss=0.3597 | top1=68.52% top5=91.65% | 1212.2s
Ep 27: train_loss=0.3441 | top1=69.54% top5=91.69% | 1213.3s
Ep 28: train_loss=0.3308 | top1=69.26% top5=91.86% | 1213.0s
Ep 29: train_loss=0.3251 | top1=69.75% top5=92.11% | 1213.7s   ← BEST
```
