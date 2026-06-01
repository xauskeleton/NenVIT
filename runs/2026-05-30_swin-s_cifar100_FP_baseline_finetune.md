# Swin-S FP fine-tune baseline — CIFAR-100 (br=0.0, reference point)

Run date: 2026-05-30
Dataset: **CIFAR-100** (torchvision, 32→224 bicubic, ImageNet-norm; head re-created cho 100 classes)
Hardware: T4 GPU (~10.4 min/epoch with AMP)
Script: `apb_fimaq/finetune.py` (full-precision, KHÔNG quantize) → `ckpt/ft_cifar100/best.pth`

## Mục đích

Đây là **reference point br=0.0** (đã đề trong README) để biết **quant cost thật** của các run APB QAT.
Trước đây các run QAT so với FP baseline zero-shot 0.72% (head random) → vô nghĩa. Số đúng để so là 90.88%.

## Summary

| Stage                       | top-1      | top-5     |
| --------------------------- | ---------- | --------- |
| Pretrained zero-shot        | 0.72%      | 4.50%     |  (head random cho 100 classes)
| **FP fine-tune best (ep 7)**| **90.88%** | 98.97%    |
| FP fine-tune final (ep 12)  | 90.11%     | 98.62%    |

## Config

```
script          = finetune.py  (FP, no quantization)
dataset         = cifar100
epochs          = 50  (early-stopped at 12, patience=5)
lr              = 1e-4
weight_decay    = 1e-4
label_smoothing = 0.1
batch_size      = 64
freeze_backbone = False  (train toàn bộ)
seed            = 3407,  AMP fp16 on
```

## Full per-epoch metrics

```
Ep  1: train_loss=1.4565 | val top1=88.90% top5=98.76% | 625.4s
Ep  2: train_loss=1.0696 | val top1=89.03% top5=98.79% | 625.1s
Ep  3: train_loss=0.9934 | val top1=89.89% top5=98.95% | 625.2s
Ep  4: train_loss=0.9503 | val top1=89.62% top5=98.80% | 624.6s   (no improve 1/5)
Ep  5: train_loss=0.9202 | val top1=90.16% top5=98.94% | 625.1s
Ep  6: train_loss=0.8966 | val top1=90.13% top5=98.75% | 624.6s   (no improve 1/5)
Ep  7: train_loss=0.8874 | val top1=90.88% top5=98.97% | 624.2s   ← BEST
Ep  8: train_loss=0.8676 | val top1=90.34% top5=98.83% | 624.7s   (1/5)
Ep  9: train_loss=0.8557 | val top1=90.63% top5=98.92% | 623.5s   (2/5)
Ep 10: train_loss=0.8527 | val top1=90.52% top5=98.82% | 622.8s   (3/5)
Ep 11: train_loss=0.8486 | val top1=89.81% top5=98.67% | 622.4s   (4/5)
Ep 12: train_loss=0.8377 | val top1=90.11% top5=98.62% | 622.0s   (5/5) → early stop
```

## Quant cost — APB QAT vs FP baseline (90.88%)

| Run | br   | scope | top-1  | **quant cost** | compression |
| --- | ---- | ----- | ------ | -------------- | ----------- |
| FP  | 0.00 | —     | 90.88% | (baseline)     | 1×          |
| C1  | 0.95 | skip  | 88.67% | **−2.21%**     | 7.6×        |
| C2  | 0.95 | full  | 87.43% | **−3.45%**     | 9.4×        |
| C3  | 0.99 | skip  | 87.19% | **−3.69%**     | 14.3×       |
| C4  | 0.99 | full  | 85.28% | **−5.60%**     | 22.7×       |

→ Quant cost thật ở mức **−2.2% đến −5.6%** tùy br/scope (KHÔNG phải "+88%" như khi so với head random).

## ⚠️ Observation quan trọng: QAT chưa init từ baseline này

`qat.py` tự load `ckpt/best.pth` (`DEFAULT_INIT_CKPT`) làm init nếu có. Nhưng baseline này lưu ở
`ckpt/ft_cifar100/best.pth` → **đường dẫn lệch**, nên 4 run C1–C4 vẫn start từ timm-pretrained +
head random (log đều hiện FP baseline 0.72%). Nếu copy:

```
cp ckpt/ft_cifar100/best.pth ckpt/best.pth
```

rồi chạy lại QAT, model sẽ init từ 90.88% thay vì head random → **post-APB cao hơn nhiều và
quant cost nhiều khả năng giảm rõ rệt**. Đây là thí nghiệm đáng chạy lại tiếp theo.
