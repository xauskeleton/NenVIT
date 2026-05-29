# Run log — APB QAT on Swin-S / Tiny ImageNet

Mỗi run lưu thành 1 file markdown gồm config + per-epoch metrics + observations.
Đặt tên: `<date>_<model>_br<ratio>_<scope>_<extra>.md`.

## Pareto so sánh các run

| # | Date       | br    | scope | DPLR λ | eff_bits | Post-APB | **Best top-1** | Note |
| - | ---------- | ----- | ----- | ------ | -------- | -------- | -------------- | ---- |
| 1 | 2026-05-28 | 0.75  | skip  | 3000   | 8.75     | 1.12%    | **85.59%**     | Baseline pipeline |
| 2 | 2026-05-29 | 0.75  | full  | 3000   | 8.75     | 1.12%    | **85.23%**     | -0.36% so với #1, +4 layers wrapped |
| 3 | 2026-05-29 | 0.95  | full  | 3000   | 2.55     | 0.85%    | **81.57%**     | -3.66% so với #2, 3.4× compression |
| 4 | _running_  | 0.99  | skip  | 3000   | 1.31     | ?        | ?              | Extreme quant test |
| 5 | _planned_  | 0.99  | skip  | OFF    | 1.31     | ?        | ?              | DPLR ablation (CE-only) |

Reference points (chưa chạy):
- **FP fine-tune baseline** (br=0.0, scope=skip, 20 ep, cùng schedule): cần để biết quant cost thực.
- **PTQ-only** (eval ngay sau APB, không QAT): cần để biết QAT đóng góp bao nhiêu.

## Cấu hình chung tất cả run

- Model: `swin_small_patch4_window7_224` (timm pretrained)
- Dataset: HF `zh-plus/tiny-imagenet`, filter 18 classes không có trong ImageNet-1k → 91k train / 9.1k val / 182 classes remap về indices ImageNet-1k.
- Optimizer: AdamW (wd=1e-4 cho weight/bias, wd=0 cho α), cosine LR schedule.
- AMP fp16 on. seed=3407. batch_size=64.
- Freeze α tại epoch=epochs/2. Mask frozen suốt training (chưa thử `--recompute-fim-every`).
- FP baseline (zero-shot, không fine-tune): top-1 = 55.75%, top-5 = 80.89%.

## Lưu ý so sánh

- **QAT > FP** trong các run là do QAT cũng fine-tune trên Tiny train (91k ảnh).
  KHÔNG phải vì quantization "tốt hơn". So sánh fair cần FP fine-tune baseline.
- Tiny ImageNet là test bed cho debug pipeline, không phải benchmark chính.
  Số kết quả chỉ valid cho relative comparison giữa các config.
