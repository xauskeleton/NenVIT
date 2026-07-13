# Swin-S — KẾT HỢP prune → quant (combined, §6.6) — init pruned g05_fisher

Run date: 2026-07-06 (Kaggle GPU) | Script: `apb_fimaq/qat.py`
**Nguồn: log Kaggle** (`/kaggle/working/checkpoints/pruned_*`) — chép về giữ (Kaggle ephemeral).
Init: `ckpt/best_pruned_g05_fisher.pt` (Swin pruned global fisher ratio0.5 → **22.0M params**, FP top1=**90.39%**).
Chung: scope full / partition **magnitude** / DPLR λ3000 / 29ep / batch32 / seed3407. `--init-model` nạp nguyên object.

> Đây là kết quả chính của đề tài (nén kép): model **đã prune** (−52.9% params) rồi **APB-QAT** (W~1bit + act-quant).
> Init pruned nên `Skipping FIM extraction (partition='magnitude')` + `[FP baseline] top1=90.39%`.

## Kết quả (best val top1)

| br | act-bit | **Best top1** | so FP 90.88% | so prune-only 90.39% | Post-APB | packed | eff-bit whole | comp. |
|----|---------|--------------|--------------|----------------------|----------|--------|---------------|-------|
| **0.95** | 2 | **83.25%** ⭐ | −7.63% | −7.14% | 1.09% | 12.1 MB | 4.39 | 9.1× |
| 0.99 | 2 | **81.19%** | −9.69% | −9.20% | 0.87% | 5.2 MB | 1.88 | 21.4× |
| 0.95 | 1 | **62.20%** | −28.68% | −28.19% | 0.92% | 12.1 MB | 4.39 | 9.1× |
| 0.99 | 1 | **52.85%** | −38.03% | −38.19%* | 1.05% | 5.2 MB | 1.88 | 21.4× |

(*52.85 − 90.39 = −37.54%; số so FP = −38.03%.) best.pth FP32 = 110.3 MB (pruned, ~½ full 244.6 MB).

## Nhận xét (§6.6)

- **Cấu hình tốt nhất: prune0.5 + br0.95 + act2 = 83.25%** — vừa nhỏ kiến trúc 52.9% (22M) vừa W~binary + act 2-bit.
- **Pruning gần "miễn phí" khi chồng lên quant:** combined br0.99/act2 = **81.19%** vs quant-only (không prune)
  br0.99/act2 magnitude+DPLR = **81.79%** → thêm nhánh prune (−52.9% params) chỉ mất **−0.60%** acc. Bằng chứng
  mạnh: 2 hướng nén **cộng hưởng**, không xung đột.
- **1-bit activation vẫn là điểm chết** (62.20 / 52.85) — nhất quán với ablation act-bit; nén kép không cứu vách đá 1-bit.
- br0.95 nén ít hơn (9.1×, 12.1 MB) nhưng acc cao hơn br0.99 (21.4×, 5.2 MB) — đánh đổi acc↔size rõ.

## Per-epoch val top1

**br0.95/act2 (83.25):** 60.87·66.82·69.03·70.62·72.21·73.85·73.89·75.01·76.16·76.50·77.32·78.13·79.00·78.39·79.02·79.37·80.80·80.68·80.87·81.33·81.61·81.75·81.77·82.03·82.42·82.67·82.60·**83.25**·82.87
**br0.99/act2 (81.19):** 50.88·59.96·62.85·65.37·67.58·69.70·70.26·70.22·72.74·73.44·74.25·74.48·75.30·76.17·76.54·77.24·78.19·78.22·78.60·78.48·79.09·79.31·80.16·80.56·80.69·80.84·81.06·**81.19**·81.08
**br0.95/act1 (62.20):** 10.93·19.19·26.87·31.27·36.64·38.88·41.30·44.41·46.40·47.92·48.83·50.33·52.20·53.48·53.73·55.33·56.80·57.15·58.23·58.74·58.60·60.17·60.30·60.85·61.48·61.71·62.18·**62.20**·61.99
**br0.99/act1 (52.85):** 11.42·17.45·21.68·24.68·28.20·30.68·33.00·33.92·36.85·38.42·40.27·41.19·43.48·44.04·44.26·46.44·46.52·48.09·48.09·48.84·49.33·50.48·50.72·50.03·51.42·51.89·52.00·51.86·**52.85**
