# Swin-S APB W+A — CIFAR-100 (br0.95 / full / act 2-bit / **fim / +DPLR λ=3000**)

Run date: 2026-06-30 (~11.5h) | Script: `apb_fimaq/qat.py` | `--partition fim --act-bits 2 --use-dplr-loss --dplr-lambda 3000`
Init-baseline (FP 90.88%). Ô fim+DPLR của bộ **br0.95 + act2**. Distillation DPLR-FIM per-block từ FP teacher.

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.95 --apb-scope full --use-dplr-loss --dplr-lambda 3000 --epochs 29 --fim-batches 10 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 2 --partition fim
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.16% | 5.20% |
| **Best (ep28)** | **80.53%** | 96.35% |

- quant cost vs FP 90.88% = **−10.35%** | eff-bit W (whole) = 4.29, packed **26.2 MB**, comp **9.3×** (Eq10 APB = 3.61 bit)
- APB: 100 layers, avg binary=0.950, avg eff_bits=3.55 | act LSQ signed 2-bit | DPLR: 24 Swin blocks, λ=3000, p1=p2=1.0
- ~1430s/ep × 29 ≈ **11.5h** (DPLR thêm FP teacher forward + per-block loss, +~17% vs no-DPLR ~1213-1228s).

## ⭐ DPLR giúp fim bao nhiêu (cùng br0.95/act2, chỉ khác DPLR)
| run | DPLR | top-1 |
|-----|------|-------|
| fim/no-DPLR | ✗ | 69.75% |
| **này** | ✓ | **80.53%** |

→ **DPLR +10.78%** cho fim ở br0.95/act2. So br0.99/act2/fim (+13.00%): DPLR vẫn giúp mạnh, hơi ít hơn.

## ⚠️ Nhưng vẫn thua magnitude/no-DPLR
| run | partition | DPLR | top-1 |
|-----|-----------|------|-------|
| magnitude | magnitude | ✗ | **82.63%** |
| **này** | fim | ✓ | 80.53% |

→ fim **có** distillation (80.53) vẫn kém magnitude **không** distillation (82.63) −2.10%. Củng cố mạnh:
**giá trị FIMA-Q ở distillation-loss chứ không ở partition** — distillation vá fim nhưng không đuổi kịp magnitude.

## So br0.99 (cùng ô fim/DPLR/act2)
| br | top-1 | Δ |
|----|-------|---|
| 0.99 (C8+A2) | 78.49% | — |
| **0.95** | **80.53%** | **+2.04%** |

## Per-epoch (đầy đủ) — DPLR (λ=3000, p1=p2=1.0)
```
Ep  1: loss=4.3774 (ce=3.4597 + λ·dplr=0.0003) | top1=34.13% top5=67.70% | 1429.9s
Ep  2: loss=2.9852 (ce=2.1139)                 | top1=53.16% top5=83.35% | 1432.6s
Ep  3: loss=2.4476 (ce=1.5994)                 | top1=60.21% top5=88.15% | 1431.2s
Ep  4: loss=2.1855 (ce=1.3532)                 | top1=62.90% top5=89.99% | 1430.5s
Ep  5: loss=2.0149 (ce=1.1964)                 | top1=66.26% top5=91.46% | 1430.0s
Ep  6: loss=1.8787 (ce=1.0719)                 | top1=67.59% top5=92.14% | 1429.6s
Ep  7: loss=1.7720 (ce=0.9770)                 | top1=69.51% top5=92.85% | 1429.0s
Ep  8: loss=1.6712 (ce=0.8866)                 | top1=71.25% top5=93.54% | 1429.7s
Ep  9: loss=1.5901 (ce=0.8169)                 | top1=71.37% top5=93.24% | 1429.5s
Ep 10: loss=1.5117 (ce=0.7474)                 | top1=72.96% top5=93.91% | 1429.7s
Ep 11: loss=1.4489 (ce=0.6930)                 | top1=74.23% top5=94.49% | 1428.9s
Ep 12: loss=1.3789 (ce=0.6299)                 | top1=74.05% top5=94.41% | 1428.9s
Ep 13: loss=1.3236 (ce=0.5823)                 | top1=74.64% top5=94.65% | 1429.3s
Ep 14: loss=1.2646 (ce=0.5295)                 | top1=75.35% top5=95.00% | 1431.2s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: loss=1.2095 (ce=0.4807)                 | top1=76.21% top5=94.97% | 1429.6s
Ep 16: loss=1.1591 (ce=0.4347)                 | top1=76.83% top5=95.11% | 1429.4s
Ep 17: loss=1.1183 (ce=0.3986)                 | top1=75.89% top5=95.15% | 1430.8s
Ep 18: loss=1.0761 (ce=0.3610)                 | top1=77.19% top5=95.45% | 1429.9s
Ep 19: loss=1.0393 (ce=0.3283)                 | top1=77.52% top5=95.86% | 1430.0s
Ep 20: loss=1.0008 (ce=0.2932)                 | top1=78.17% top5=95.91% | 1430.2s
Ep 21: loss=0.9744 (ce=0.2706)                 | top1=79.00% top5=96.12% | 1428.6s
Ep 22: loss=0.9396 (ce=0.2384)                 | top1=79.48% top5=96.14% | 1432.0s
Ep 23: loss=0.9172 (ce=0.2191)                 | top1=79.54% top5=96.24% | 1430.8s
Ep 24: loss=0.8979 (ce=0.2025)                 | top1=79.25% top5=96.17% | 1429.9s
Ep 25: loss=0.8774 (ce=0.1837)                 | top1=79.79% top5=96.09% | 1428.3s
Ep 26: loss=0.8637 (ce=0.1716)                 | top1=80.14% top5=96.30% | 1430.1s
Ep 27: loss=0.8511 (ce=0.1597)                 | top1=80.08% top5=96.22% | 1430.5s
Ep 28: loss=0.8456 (ce=0.1548)                 | top1=80.53% top5=96.35% | 1431.0s   ← BEST
Ep 29: loss=0.8372 (ce=0.1468)                 | top1=80.29% top5=96.29% | 1432.2s
```
(⚠️ nhãn `λ·dplr` in giá trị dplr thô ~2-3e-4, KHÔNG nhân λ; loss tổng đã cộng đúng λ·dplr — TODO sửa nhãn.)
