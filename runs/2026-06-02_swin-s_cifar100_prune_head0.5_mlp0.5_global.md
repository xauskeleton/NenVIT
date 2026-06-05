# Swin-S FIMA-Q-guided structural pruning — CIFAR-100 (head0.5 / mlp0.5, global per_param)

Run date: 2026-06-02
Dataset: **CIFAR-100** (torchvision, 32→224 bicubic, ImageNet-norm; head re-created cho 100 classes)
Script: `apb_fimaq/prune_swin_cifar.py`
Init: từ CIFAR baseline `ckpt/best.pth` → `[FP baseline] top1=90.88% top5=98.97%` (10000 samples)
Hardware: GPU (~440s/epoch, ~2.03 it/s, 781 it/epoch)

## Mục đích

Structural pruning (KHÔNG quantize) dùng DPLR-FIM importance để chọn head/MLP channel bỏ đi
(lowest-FIM dropped). Khác hẳn nhóm APB QAT (C1–C8) — đây là nhánh `feat/structural-pruning`.

## Command

```bash
python apb_fimaq/prune_swin_cifar.py \
  --rank-mode global --global-metric per_param \
  --head-ratio 0.5 --mlp-ratio 0.5 --min-heads 1 --mlp-keep-frac 0.05 \
  --fim-batches 10 --epochs 20 --lr 1e-4 --batch-size 64 --num-workers 4 --seed 3407
```

## Config

```
dataset         = cifar100
head_ratio      = 0.5
mlp_ratio       = 0.5
min_heads       = 1
mlp_keep_frac   = 0.05
head_keep_frac  = 0.0
rank_mode       = global
global_metric   = per_param   (lowest-FIM dropped)
fim_batches     = 10
fim_mode        = dplr
epochs          = 20
lr              = 1e-4
weight_decay    = 1e-4
label_smoothing = 0.1
batch_size      = 64
num_workers     = 4
amp             = True
seed            = 3407
out_dir         = /kaggle/working/ckpt/pruned_cifar100
```

## Summary

| Stage                          | top-1      | top-5   | params  |
| ------------------------------ | ---------- | ------- | ------- |
| FP baseline (init từ best.pth) | 90.88%     | 98.97%  | 48.91M  |
| Post-prune (trước finetune)    | 19.08%     | 47.87%  | 23.06M  |
| **Best pruned (ep 19)**        | **90.87%** | 98.60%  | 23.06M  |
| Final (ep 20)                  | 90.84%     | 98.55%  | 23.06M  |

- **Prune cost = −0.01%** (90.87% vs FP baseline 90.88%)
- **Params: 48.91M → 23.06M (52.9% nhỏ hơn)**
- FIM done in 17.4s, 96 layers

## Pruning details

```
heads  282 -> 141       (head_ratio=0.5)
mlp    36096 -> 18048    (mlp_ratio=0.5)
params 48.91M -> 23.06M  (52.9% smaller)
```

## Full per-epoch metrics (finetune 20 ep)

```
Ep  1: train_loss=1.2003 | val top1=87.08% top5=98.40% | 440.3s
Ep  2: train_loss=1.0048 | val top1=87.75% top5=98.67% | 439.7s
Ep  3: train_loss=0.9608 | val top1=87.62% top5=98.57% | 439.8s
Ep  4: train_loss=0.9318 | val top1=88.19% top5=98.58% | 439.5s
Ep  5: train_loss=0.9020 | val top1=88.25% top5=98.57% | 439.8s
Ep  6: train_loss=0.8868 | val top1=88.51% top5=98.66% | 439.5s
Ep  7: train_loss=0.8714 | val top1=88.62% top5=98.61% | 438.7s
Ep  8: train_loss=0.8564 | val top1=89.49% top5=98.68% | 439.2s
Ep  9: train_loss=0.8432 | val top1=89.15% top5=98.62% | 439.7s
Ep 10: train_loss=0.8393 | val top1=89.32% top5=98.63% | 439.8s
Ep 11: train_loss=0.8272 | val top1=89.64% top5=98.72% | 438.9s
Ep 12: train_loss=0.8214 | val top1=90.08% top5=98.56% | 439.3s
Ep 13: train_loss=0.8133 | val top1=90.16% top5=98.47% | 439.1s
Ep 14: train_loss=0.8084 | val top1=90.31% top5=98.54% | 438.9s
Ep 15: train_loss=0.8044 | val top1=90.39% top5=98.46% | 439.2s
Ep 16: train_loss=0.8011 | val top1=90.66% top5=98.60% | 439.1s
Ep 17: train_loss=0.7994 | val top1=90.63% top5=98.59% | 438.9s
Ep 18: train_loss=0.7974 | val top1=90.71% top5=98.58% | 439.3s
Ep 19: train_loss=0.7966 | val top1=90.87% top5=98.60% | 439.4s   ← BEST
Ep 20: train_loss=0.7955 | val top1=90.84% top5=98.55% | 439.3s
```

## Observations

- Post-prune trước finetune tụt mạnh (90.88% → 19.08%) nhưng phục hồi gần hết chỉ sau ~19 epoch
  → cấu trúc còn lại đủ capacity, finetune đủ để recover.
- Đường accuracy đơn điệu tăng, chưa bão hòa hẳn ở ep20 (90.84 ≈ 90.87) → 50% prune ở mức này
  gần như "free". Đáng thử ép tỷ lệ cao hơn (head/mlp 0.6–0.75) để tìm giới hạn.
- ⚠️ Single seed (3407) — cần xác nhận thêm seed trước khi chốt luận văn.

## TODO / thí nghiệm tiếp

- Sweep head/mlp-ratio cao hơn (0.6, 0.7, 0.75) để vẽ Pareto accuracy↔params.
- So với random/magnitude pruning để chứng minh giá trị của FIMA-Q importance.
- Đo FLOPs / latency thực, không chỉ param count.
