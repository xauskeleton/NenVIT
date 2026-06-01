# Run log — APB QAT on Swin-S / Tiny ImageNet

Mỗi run lưu thành 1 file markdown gồm config + per-epoch metrics + observations.
Đặt tên: `<date>_<model>_br<ratio>_<scope>_<extra>.md`.

## Pareto so sánh các run

| # | Date       | br    | scope | DPLR λ | eff_bits | Post-APB | **Best top-1** | Note |
| - | ---------- | ----- | ----- | ------ | -------- | -------- | -------------- | ---- |
| 1 | 2026-05-28 | 0.75  | skip  | 3000   | 8.75     | 1.12%    | **85.59%**     | Baseline pipeline |
| 2 | 2026-05-29 | 0.75  | full  | 3000   | 8.75     | 1.12%    | **85.23%**     | -0.36% so với #1, +4 layers wrapped |
| 3 | 2026-05-29 | 0.95  | full  | 3000   | 2.55     | 0.85%    | **81.57%**     | -3.66% so với #2, 3.4× compression |
| 4 | → CIFAR   | 0.99  | skip  | 3000   | —        | —        | _xem C6_       | Đã chạy trên **CIFAR-100** (89.32%), không phải Tiny → xem bảng C6 |
| 5 | → CIFAR   | 0.99  | skip  | OFF    | —        | —        | _xem C5_       | Đã chạy trên **CIFAR-100** (88.06%), không phải Tiny → xem bảng C5 |

Reference points (chưa chạy):
- **FP fine-tune baseline** (br=0.0, scope=skip, 20 ep, cùng schedule): cần để biết quant cost thực.
- **PTQ-only** (eval ngay sau APB, không QAT): cần để biết QAT đóng góp bao nhiêu.

## CIFAR-100 (Swin-S) — dataset khác, KHÔNG so trực tiếp với bảng Tiny ImageNet ở trên

torchvision CIFAR-100, 32→224 bicubic, head re-created cho 100 classes.
fim_batches=10, lr=1e-4, bs=64, AMP, seed=3407. Epochs: C1–C4 = 20ep (head-random), C5–C6 = 30ep (init-baseline).
**FP fine-tune baseline (br=0.0, finetune.py, 12 ep early-stop): top-1 = 90.88%** ← mốc so quant cost.

| # | Date       | br   | scope | DPLR λ | **eff bit (whole)** | Post-APB | **Best top-1** | quant cost | **FP32 gốc (best.pth)** | packed | comp. | Note |
| - | ---------- | ---- | ----- | ------ | ------------------- | -------- | -------------- | ---------- | ----------------------- | ------ | ----- | ---- |
| FP | 2026-05-30 | 0.00 | —    | —      | 32.0                | —        | **90.88%**     | (baseline) | —                       | —       | 1×    | FP fine-tune reference |
| C1 | 2026-05-30 | 0.95 | skip  | 3000   | **5.20**            | 0.60%    | **88.67%**     | −2.21%     | ~241.7 MB†              | 31.8 MB | 7.6×  | Accuracy cao nhất |
| C2 | 2026-05-30 | 0.95 | full  | 3000   | **4.27**            | 1.13%    | **87.43%**     | −3.45%     | ~245.3 MB†              | 26.1 MB | 9.4×  | −1.24% vs C1, file nhỏ hơn 5.7 MB |
| C3 | 2026-05-31 | 0.99 | skip  | 3000   | **2.78**            | 0.92%    | **87.19%**     | −3.69%     | 242.9 MB                | 17.0 MB | 14.3× | head-random, 20ep. Sweet spot cũ |
| C4 | 2026-05-31 | 0.99 | full  | 3000   | **1.77**            | 0.97%    | **85.28%**     | −5.60%     | ~245.2 MB†              | 10.8 MB | 22.7× | head-random, 20ep. Nén cao nhất |
| **C5** | 2026-05-31 | 0.99 | skip | OFF (CE) | **2.78**          | 1.84%    | **88.06%**     | −2.82%     | 242.9 MB                | 17.0 MB | 14.3× | ✅ init-baseline, 30ep, **CE-only** |
| **C6** | 2026-05-31 | 0.99 | skip | 3000   | **2.78**            | 1.84%    | **89.32%**     | **−1.56%** | 242.9 MB                | 17.0 MB | 14.3× | ✅ init-baseline, 30ep, **DPLR**. Accuracy cao nhất |
| **C7** | 2026-05-31 | 0.99 | full | OFF (CE) | **1.77**          | 1.54%    | **87.44%**     | −3.44%     | 244.6 MB                | 10.8 MB | 22.7× | ✅ init-baseline, 30ep, **CE-only** |
| **C8** | 2026-05-31 | 0.99 | full | 3000   | **1.77**            | 1.54%    | **88.55%**     | **−2.33%** | 244.6 MB                | 10.8 MB | 22.7× | ✅ init-baseline, 30ep, **DPLR**. File nhỏ nhất |

† `best.pth` của C1/C2/C4 **suy ra** từ `packed × comp` (logs cũ không in trực tiếp). C3/C5/C6/C7/C8 là số in thật.
`best.pth` = checkpoint FP32 (gồm latent_weight APB + α + buffer, ~243 MB) — **mẫu số chuẩn của compression** (theo logs). Compression = best.pth ÷ packed.

(eff bit = whole-model thật = file×8 ÷ 48.91M params; số Eq10 paper-compare ở bảng "Effective bit" bên dưới.)

✅ **C5 & C6 đã fix init-from-baseline** (copy `ckpt/ft_cifar100/best.pth` → `ckpt/best.pth`; log xác nhận
`[FP baseline] top1=90.88%`). C1–C4 cũ vẫn start head-random → KHÔNG đo được quant cost thật; chỉ C5/C6
có quant cost fair (so với baseline 90.88%).

**Ma trận 2×2 init-baseline (br0.99, 30ep, seed3407) — scope × DPLR loss:**

| Best top-1 | CE-only | DPLR λ=3000 | **DPLR giúp** |
| ---------- | ------- | ----------- | ------------- |
| **skip** (17 MB, 14.3×) | C5: 88.06% | C6: **89.32%** | **+1.26%** |
| **full** (10.8 MB, 22.7×) | C7: 87.44% | C8: **88.55%** | **+1.11%** |
| **skip > full** | +0.62% | +0.77% | |

→ **2 kết luận fair (cùng init-baseline):**
1. **DPLR loss luôn thắng CE-only** ở cả 2 scope (+1.26% skip, +1.11% full). Đảo ngược note cũ
   "DPLR không phải driver" (note cũ dựa trên runs head-random). Magnitude loss vẫn ~1e-4 nhưng
   hướng gradient per-block khớp FP teacher giúp generalize. Đổi lại chậm hơn (~13.4 vs ~9.5 min/ep).
2. **skip > full nhưng gap hẹp** (+0.62~0.77%) — hẹp hơn nhiều so với head-random (C3>C4 = −1.91%).
   init-baseline làm full đỡ thiệt khi quantize cả head/downsample. full đổi <0.8% accuracy lấy file
   nhỏ hơn 37% (10.8 vs 17 MB) + nén 22.7× (vs 14.3×).

⚠️ Tất cả single seed — nên xác nhận thêm 1–2 seed trước khi chốt vào luận văn.

### Effective bit (whole-model = số chính)

Tổng params Swin-S (num_classes=100) = **48,914,158 ≈ 48.91M**. Effective bit = file `.pt` thật × 8 ÷ tổng params = **số bit trung bình/weight thực sự khi deploy**.

| Run | **eff bit (whole-model)** | (paper-compare, Eq10) |
| --- | ------------------------- | --------------------- |
| C1 (br0.95 skip) | **5.20** | 3.61 |
| C2 (br0.95 full) | **4.27** | 3.61 |
| C3 (br0.99 skip) | **2.78** | 1.52 |
| C4 (br0.99 full) | **1.77** | 1.52 |
| C5 (br0.99 skip, CE-only) | **2.78** | 1.52 |
| C6 (br0.99 skip, DPLR)    | **2.78** | 1.52 |
| C7 (br0.99 full, CE-only) | **1.77** | 1.52 |
| C8 (br0.99 full, DPLR)    | **1.77** | 1.52 |

- **eff bit (whole-model)** = số dùng chính (deploy/luận văn). Gồm cả `_other` fp32 (LN, bias, head; +downsample nếu scope=skip) → đổi theo scope.
- Cột **(paper-compare)** = công thức Eq 10 chỉ-vùng-APB, b_p-bit positions — CHỈ để so bảng paper FIMA-Q/APB (paper không tính phần fp32 ngoài layer quantize). Đừng lấy whole-model đi so paper.
- eff bit là bit **trung bình/weight** (weight thật là 1-bit binary hoặc ~52-bit FP, không phải mỗi cái 5.2 bit).

**⚠️ Bảng trộn 2 chế độ init — KHÔNG so chéo head-random ↔ init-baseline:**
- **C1–C4 = head-random** (head ngẫu nhiên, 20ep) → quant cost KHÔNG fair.
- **C5–C6 = init-baseline 90.88%** (30ep) → quant cost fair.
- C6 (89.32%) > C1 (88.67%) **KHÔNG** kết luận được "br0.99 > br0.95": C6 có lợi thế init-baseline mà C1 không có. Muốn so công bằng phải re-run br0.95 skip/full với init-baseline.

**Pareto frontier FAIR** (chỉ xét 4 run init-baseline C5–C8, non-dominated): **C6 và C8**.
- **C6** (skip, DPLR): acc cao nhất **89.32%** @ 2.78 bit, 17 MB, 14.3× → chọn khi ưu tiên accuracy.
- **C8** (full, DPLR): file nhỏ nhất **10.8 MB** @ 1.77 bit, 22.7× @ 88.55% → chọn khi ưu tiên storage.
- C5 (skip CE) bị C6 dominate; C7 (full CE) bị C8 dominate → **DPLR thắng CE ở cả 2 scope**.

⚠️ C1–C4 (head-random, br0.95 + br0.99) **không cùng frontier** với C5–C8 vì khác init. Muốn so
br0.95 vs br0.99 công bằng phải re-run **br0.95 skip/full + init-baseline** (thí nghiệm còn thiếu).
- skip > full (nhóm head-random: +1.24% @0.95 C1>C2, +1.91% @0.99 C3>C4; nhóm init-baseline: +0.62~0.77%).
- Tăng br 0.95→0.99 (nhóm head-random): skip −1.48% (C1→C3), full −2.15% (C2→C4); nén gần gấp đôi.
- ⚠️ eff bit whole-model còn cao hơn paper-compare khá nhiều (C3: 2.78 vs 1.52) do int32 positions
  + downsample/head fp32. Sửa pack (b_p/bitmap + nén downsample) sẽ kéo whole-model về gần Eq10.
  `actual_effective_bits` trong qat.py tự in whole-model mỗi run.

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
