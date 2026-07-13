# Swin-S APB — A2 partition ablation: **fisher** vs magnitude (br0.99 / full / act2)

Run date: 2026-07-03 → 2026-07-04 (Kaggle GPU) | Script: `apb_fimaq/qat.py`
**Nguồn: log Kaggle** (`/kaggle/working/checkpoints/...`) — chép về đây để giữ (Kaggle ephemeral).
Init: FP baseline `best.pth` (top1=90.88%). Chung: br0.99 / scope full / act-bits 2 (LSQ) / 29ep / batch32 / seed3407 / fim_batches10.

> ⭐ **Đây là A2 "đúng" — partition FIM dùng `fisher` chuẩn** (`--partition fim --importance fisher`), thay cho
> bản `dplr` tự chế đã deprecated. Trả lời câu hỏi: *fisher-partition có thắng magnitude không?* → **KHÔNG.**

## Kết quả (best val top1)

| partition | DPLR λ3000 | **Best top1** | out-dir (Kaggle) |
|-----------|-----------|--------------|------------------|
| **fisher** | — | **65.78%** | `br0.99_act2_fisher` |
| **fisher** | ✅ | **77.80%** | `br0.99_act2_fisher_dplr` |

Post-APB (trước QAT) ~1.1%. eff-bit whole 1.78, packed 10.8 MB, 22.5×.
*(Các biến thí fisher+LR đã bỏ — LR không thuộc method.)*

## ⭐ So sánh partition (cùng br0.99/act2/29ep/batch32/seed) — A2 KHÉP

| partition | no-DPLR | +DPLR λ3000 |
|-----------|---------|-------------|
| **magnitude ⭐** | **77.98** | **81.79** |
| **fisher** | 65.78 | 77.80 |
| **gap (mag − fisher)** | **+12.20** | **+3.99** |

## Kết luận A2 (fisher vs magnitude)

1. **magnitude THẮNG fisher** cả 2 cột (+12.20% no-DPLR, +3.99% +DPLR) → **kết quả âm cho "FIM-guided partition",
   chứng minh với fisher chuẩn.** Nguyên nhân: giữ |w| lớn FP giảm trực tiếp sai số APB `α·sign(w)`, còn FIM
   (curvature) chọn outlier ít liên quan tới sai số tái tạo → thua.
2. ⇒ **Giá trị FIMA-Q nằm ở distillation loss (DPLR), KHÔNG ở partition.** Config tốt nhất = **magnitude + DPLR**.

## Per-epoch val top1 (2 run sạch)

**fisher / no-DPLR (65.78):** 14.78 · 26.18 · 34.39 · 40.09 · 44.06 · 48.67 · 50.92 · 52.98 · 53.69 · 54.93 · 58.25 · 58.12 · 58.36 · 59.94 · 59.56 · 60.82 · 62.47 · 62.86 · 63.32 · 63.55 · 63.42 · 64.69 · 64.66 · 65.29 · 64.95 · 65.29 · 65.58 · 65.49 · **65.78**

**fisher / +DPLR (77.80):** 22.53 · 40.64 · 50.62 · 55.19 · 59.10 · 63.67 · 64.41 · 66.18 · 67.55 · 69.09 · 70.60 · 70.88 · 70.93 · 71.35 · 72.22 · 73.32 · 73.73 · 74.42 · 74.86 · 75.00 · 76.18 · 75.87 · 76.17 · 76.56 · 77.52 · 77.28 · 77.27 · 77.52 · **77.80**
