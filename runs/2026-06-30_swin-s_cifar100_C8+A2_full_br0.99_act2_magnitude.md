# Swin-S APB W+A — CIFAR-100 (C8+A2 **MAGNITUDE / DPLR** / full / br0.99 / act 2-bit)

Run date: 2026-06-30 (~11.7h) | Script: `apb_fimaq/qat.py` | `--partition magnitude --act-bits 2 --use-dplr-loss --dplr-lambda 3000`
Khép cặp sạch act2: cùng C8+A2 fim+DPLR (78.49) chỉ khác `--partition` ⇒ "magnitude > fim ở act2 (DPLR)?".
Đây cũng là ô cuối khép **2×2 partition × DPLR ở act2** (3 ô kia đã có).

## Command
```bash
python apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --apb-scope full --use-dplr-loss --dplr-lambda 3000 --epochs 29 --fim-batches 10 --lr 1e-4 --batch-size 32 --num-workers 4 --seed 3407 --act-bits 2 --partition magnitude
```

## Summary
| Stage | top-1 | top-5 |
| ----- | ----- | ----- |
| FP baseline | 90.88% | 98.97% |
| Post-APB (trước QAT) | 1.12% | 5.37% |
| **Best (ep29)** | **81.79%** | 96.82% |

- quant cost vs FP 90.88% = **−9.09%** | eff-bit W = 1.78, packed **10.8 MB**, comp **22.5×**
- APB: 100 layers, avg binary=0.990, avg eff_bits=1.51 | act LSQ signed 2-bit trên 100 APB-Linear inputs
- ~1452s/ep × 29 ≈ **11.7h**. val vẫn nhích đều tới ep29 (81.3→81.8, **chưa plateau**); train ce thấp (0.16).

## ⭐ Cặp SẠCH với C8+A2 (fim+DPLR) — chỉ khác partition
| run | act | partition | DPLR | top-1 |
|-----|-----|-----------|------|-------|
| C8+A2 (fim) | 2 | **fim** | ✓ | 78.49% |
| **này** | 2 | **magnitude** | ✓ | **81.79%** |

→ **magnitude > fim ở act2 CÓ DPLR = +3.30%** (cặp sạch tuyệt đối, chỉ khác partition). Khẳng định lại
magnitude thắng fim, lần này ở chế độ DPLR/act2 (trước đó đã có no-DPLR/act2 = +12.49% và cả 2 cột act1).

## ⭐⭐ 2×2 ĐẦY ĐỦ partition × DPLR ở **act2** (full / br0.99 / 29ep / batch32 / seed3407)
| partition \ DPLR | **no-DPLR** | **DPLR** | → DPLR giúp |
|------------------|-------------|----------|-------------|
| **fim** | 65.49% | 78.49% | **+13.00%** |
| **magnitude** | 77.98% | **81.79%** | +3.81% |
| **→ magnitude − fim** | **+12.49%** | **+3.30%** | |

**Kết luận (2×2 act2 đầy đủ):**
1. **magnitude > fim ở CẢ 4 ô** (act1 lẫn act2, no-DPLR lẫn DPLR) → finding cực kỳ robust. fim-cho-partition
   thua magnitude ở mọi nơi đã đo.
2. **Tương tác DPLR×partition ĐẢO CHIỀU giữa act1 và act2:**
   - act1: DPLR giúp **magnitude** nhiều hơn (+9.42 vs +5.93) → gap **NỚI RỘNG** 3.77 → 7.26.
   - act2: DPLR giúp **fim** nhiều hơn (+13.00 vs +3.81) → gap **CO LẠI** 12.49 → 3.30.
   → Ở act2, DPLR (distillation dùng FIM) **bù phần lớn** cho partition tệ của fim, kéo fim 65.49→78.49 gần
   magnitude+DPLR. Magnitude vốn đã tốt nên DPLR ít chỗ để cải thiện (ceiling effect, +3.81).
3. **magnitude+DPLR là tốt nhất ở act2 (81.79%)** — đúng như act1 (magnitude+DPLR=58.29 cũng tốt nhất).
4. **fim KHÔNG lật lại magnitude khi act tăng 1→2** kể cả khi bật DPLR. Giả thuyết "fim cần activation tốt
   hơn để thắng" tiếp tục bị bác bỏ.

## So sánh act-bit (magnitude + DPLR, cùng partition + cùng DPLR)
| act | top-1 | file |
|-----|-------|------|
| 1 | 58.29% | `..._act1_magnitude.md` |
| **2** | **81.79%** | file này |

→ act1→act2 (magnitude, DPLR) = **+23.50%**. Vẫn vách đá act-bit lớn nhưng nhỏ hơn no-DPLR (+29.11%):
DPLR nâng sàn act1 nhiều hơn nên khoảng cách act1↔act2 hẹp lại.

## ⚡ Tốc độ
~1452s/ep (LSQ 2-bit + DPLR) rất ổn định (1442-1466s). So act2 magnitude **no-DPLR** ~1202s/ep → DPLR thêm
~250s/ep (~+21%, khớp pattern "DPLR thêm forward teacher + per-block loss"). 29ep ≈ 11.7h.

## Per-epoch (đầy đủ) — DPLR (λ=3000, p1=p2=1.0)
```
Ep  1: loss=4.4745 (ce=3.0365 + λ·dplr=0.0005) | top1=47.72% top5=80.01% | 1442.5s
Ep  2: loss=3.0463 (ce=1.7317)                 | top1=58.09% top5=86.77% | 1456.5s
Ep  3: loss=2.6528 (ce=1.3929)                 | top1=64.28% top5=90.37% | 1453.6s
Ep  4: loss=2.4238 (ce=1.2042)                 | top1=66.32% top5=91.65% | 1443.9s
Ep  5: loss=2.2631 (ce=1.0775)                 | top1=69.04% top5=92.67% | 1465.4s
Ep  6: loss=2.1338 (ce=0.9768)                 | top1=70.02% top5=93.08% | 1443.5s
Ep  7: loss=2.0295 (ce=0.8960)                 | top1=72.33% top5=93.55% | 1459.4s
Ep  8: loss=1.9323 (ce=0.8227)                 | top1=72.74% top5=93.85% | 1450.7s
Ep  9: loss=1.8527 (ce=0.7622)                 | top1=74.00% top5=94.66% | 1444.7s
Ep 10: loss=1.7778 (ce=0.7035)                 | top1=74.49% top5=94.54% | 1464.8s
Ep 11: loss=1.7066 (ce=0.6483)                 | top1=76.04% top5=95.11% | 1444.1s
Ep 12: loss=1.6463 (ce=0.6010)                 | top1=75.67% top5=95.14% | 1459.7s
Ep 13: loss=1.5865 (ce=0.5544)                 | top1=76.43% top5=95.44% | 1449.2s
Ep 14: loss=1.5405 (ce=0.5195)                 | top1=77.01% top5=95.80% | 1444.1s
  >> Epoch 15: α frozen (latent_weight still trainable)
Ep 15: loss=1.4796 (ce=0.4691)                 | top1=77.63% top5=95.51% | 1462.8s
Ep 16: loss=1.4271 (ce=0.4268)                 | top1=78.27% top5=96.06% | 1443.1s
Ep 17: loss=1.3842 (ce=0.3941)                 | top1=78.58% top5=96.10% | 1463.7s
Ep 18: loss=1.3454 (ce=0.3631)                 | top1=79.07% top5=95.99% | 1444.9s
Ep 19: loss=1.3033 (ce=0.3299)                 | top1=79.36% top5=96.27% | 1443.2s
Ep 20: loss=1.2661 (ce=0.2987)                 | top1=79.63% top5=96.24% | 1465.4s
Ep 21: loss=1.2338 (ce=0.2741)                 | top1=80.20% top5=96.39% | 1443.0s
Ep 22: loss=1.2001 (ce=0.2453)                 | top1=80.96% top5=96.50% | 1465.9s
Ep 23: loss=1.1798 (ce=0.2305)                 | top1=80.85% top5=96.45% | 1443.3s
Ep 24: loss=1.1531 (ce=0.2094)                 | top1=80.97% top5=96.59% | 1446.3s
Ep 25: loss=1.1364 (ce=0.1966)                 | top1=81.31% top5=96.55% | 1463.1s
Ep 26: loss=1.1188 (ce=0.1818)                 | top1=81.38% top5=96.79% | 1444.3s
Ep 27: loss=1.1090 (ce=0.1738)                 | top1=81.25% top5=96.82% | 1465.1s
Ep 28: loss=1.0984 (ce=0.1648)                 | top1=81.54% top5=96.86% | 1443.8s
Ep 29: loss=1.0905 (ce=0.1580)                 | top1=81.79% top5=96.82% | 1445.8s   ← BEST
```
(⚠️ nhãn `λ·dplr` in giá trị dplr thô ~3-5e-4, KHÔNG nhân λ; loss tổng đã cộng đúng λ·dplr — TODO sửa nhãn.)
