# Run log — APB QAT on Swin-S / Tiny ImageNet

Mỗi run lưu thành 1 file markdown gồm config + per-epoch metrics + observations.
Đặt tên: `<date>_<model>_br<ratio>_<scope>_<extra>.md`.

---

# Structural pruning (Swin-S / CIFAR-100) — nhánh `feat/structural-pruning`

Khác hẳn nhóm APB QAT (C1–C8 bên dưới): đây là **structural pruning** (bỏ head/MLP channel theo
DPLR-FIM importance, lowest-FIM dropped), KHÔNG quantize. Script `apb_fimaq/prune_swin_cifar.py`.
Init từ CIFAR baseline `ckpt/best.pth` → **FP baseline = 90.88%** (mốc so prune cost).
Chung: CIFAR-100, fim_batches=10, lr=1e-4, bs=64, AMP, seed=3407, wd=1e-4, label_smoothing=0.1.

Tất cả: 20 ep, post-prune top-1 = trước finetune (xem file run). prune cost = best top-1 − 90.88%.

| #  | mode       | head/mlp | heads     | mlp ch.        | params (M)            | post-prune | **Best top-1** | prune cost  | file |
| -- | ---------- | -------- | --------- | -------------- | --------------------- | ---------- | -------------- | ----------- | ---- |
| FP | —          | —        | 282       | 36096          | 48.91                 | —          | **90.88%**     | (baseline)  | `2026-05-30_..._FP_baseline_finetune.md` |
| P1 | **global** | 0.5      | 282 → 141 | 36096 → 18048  | **23.06** (−52.9%)    | 19.08%     | **90.87%**     | **−0.01%**  | `..._prune_head0.5_mlp0.5_global.md` |
| P2 | per_layer  | 0.5      | 282 → 142 | 36096 → 18048  | **25.32** (−48.2%)    | 7.01%      | **90.55%**     | −0.33%      | `..._prune_head0.5_mlp0.5_perlayer.md` |
| P3 | **global** | 0.7      | 282 → 85  | 36096 → 10829  | **14.27** (−70.8%)    | 2.05%      | **87.80%**     | −3.08%      | `..._prune_head0.7_mlp0.7_global.md` |
| P4 | per_layer  | 0.7      | 282 → 92  | 36096 → 10832  | **16.19** (−66.9%)    | 1.30%      | **87.85%**     | −3.03%      | `..._prune_head0.7_mlp0.7_perlayer.md` |
| P5 | **global** | 0.9      | 282 → 28  | 36096 → 3610   | **5.97** (−87.8%)     | 1.03%      | **63.00%**     | −27.88%     | `..._prune_head0.9_mlp0.9_global.md` |
| P6 | per_layer  | 0.9      | 282 → 26  | 36096 → 3616   | **6.20** (−87.3%)     | 1.00%      | **60.64%**     | −30.24%     | `..._prune_head0.9_mlp0.9_perlayer.md` |

(mọi run: rank-mode global dùng `--global-metric per_param --mlp-keep-frac 0.05`; FIM=DPLR, seed 3407.)

**So sánh global vs per_layer (cùng ratio):**

| ratio | global (acc / params)   | per_layer (acc / params) | → kết luận |
| ----- | ----------------------- | ------------------------ | ---------- |
| 0.5   | **90.87% / 23.06M**     | 90.55% / 25.32M          | global thắng **cả 2** (+0.32% acc, ít hơn 2.3M params) |
| 0.7   | 87.80% / **14.27M**     | **87.85%** / 16.19M      | acc ~hòa (per_layer +0.05%) nhưng global ít hơn ~2M params (−12%) |
| 0.9   | **63.00% / 5.97M**      | 60.64% / 6.20M           | global thắng **cả 2** (+2.36% acc, ít params hơn) |

→ **global ≥ per_layer ở mọi mức**: cùng accuracy nhưng ít params hơn (pool toàn cục tự phân bổ cắt
nhiều ở block "rẻ"). Khoảng cách rõ nhất ở ratio cao (0.9: +2.36%). **Khuyến nghị dùng global.**

**Observations / Pareto pruning:**
- **Sweet spot = 0.5–0.7.** P1 (global 0.5) gần như free (−0.01% @ −52.9%). P3 (global 0.7) đổi
  −3.08% acc lấy −70.8% params — vẫn rất tốt cho thesis.
- **0.9 sụp đổ.** Post-prune ~1% (≈ random 100-class), 20 ep KHÔNG recover hết (val vẫn đang tăng ở
  ep20: P5 63.0%, P6 60.6%). Cắt 6 head/3.6k MLP còn lại quá ít capacity. Nếu cần ép 0.9 thì phải
  nhiều epoch hơn — nhưng cost (−28~30%) quá đắt, không đáng.
- **Pareto frontier (fair):** P1 (90.87%/23M) → P3 (87.80%/14.3M) → P5 (63%/6M). Cả 3 đều là **global**;
  per_layer P2/P6 bị global dominate, P4 ngang P3 nhưng nhiều params hơn.
- Post-prune (trước finetune) tụt mạnh hơn khi ratio cao & per_layer (P2 7.01% < P1 19.08%) nhưng
  finetune san bằng phần lớn ở ratio ≤0.7.

## ⭐ Ablation IMPORTANCE cho pruning (dplr / fisher / magnitude) — global, seed 3407

So các importance metric ở CÙNG config global (`--global-metric per_param --mlp-keep-frac 0.05`),
đổi head/mlp ratio 0.5/0.7/0.9. Mỗi ô = **best top-1 % (params M)** sau finetune. FP baseline 90.88%.
(fisher+LR+full = `--importance fisher --logits-reversal --importance-full`.)

| importance         | 0.5              | 0.7              | 0.9              |
| ------------------ | ---------------- | ---------------- | ---------------- |
| **dplr**           | **90.87** (23.1) | **87.80** (14.3) | 63.00 (6.0)      |
| **fisher**         | 90.39 (22.1)     | 87.50 (13.6)     | 71.07 (5.7)      |
| **fisher+LR+full** | 90.40 (23.4)     | 87.47 (14.2)     | **72.26** (5.7)  |
| magnitude l2sq     | 88.33 (25.8)     | 79.01 (15.8)     | 53.08 (5.9)      |
| magnitude l1       | 87.60 (26.1)     | 76.71 (16.0)     | 51.31 (5.9)      |

**Kết luận (nguồn: `image.png`):**
1. **⭐ FIM (dplr/fisher) THẮNG magnitude ở MỌI ratio** — NGƯỢC với ablation partition của QAT (magnitude
   thắng fim 9/9 ô). ⇒ FIMA-Q importance **có giá trị cho structural pruning** (dù không cho partition).
   Trả lời TODO "chứng minh giá trị FIMA-Q importance": ✅ đã chứng minh (cho pruning).
2. **dplr ≈ fisher ở 0.5/0.7** (dplr acc nhỉnh +0.3%, fisher ít params hơn ~1M) → Pareto xấp xỉ ngang.
3. **⭐ Ở 0.9, fisher vượt trội dplr: 71–72% vs 63% (+8~9%) mà ít params hơn** (5.7 vs 6.0M) → fisher
   BỀN hơn khi prune cực hạn. `+LR+full` nhích thêm chút (72.26).
4. **magnitude l2sq > l1** nhất quán; cả hai bị dominate → đúng vai trò baseline.

⚠️ Checkpoint các run trên ĐÃ BỊ ĐÈ (mọi run ghi cùng `--out-dir` mặc định `ckpt/pruned_cifar100`, tên
file cố định). Bản còn giữ riêng: `ckpt/best_pruned_g05_fisher.pt` (fisher 0.5, 22M) — dùng bản này để
quantize (prune+quant). Re-run bản khác PHẢI truyền `--out-dir` riêng.

**TODO pruning:**
- ratio trung gian 0.6 (global) để fill khoảng 23M↔14M.
- 0.9 với nhiều epoch hơn (30–40) xem có recover không (giá trị học thuật, chứng minh capacity floor).
- ✅ Baseline magnitude pruning ĐÃ CHẠY (bảng importance ở trên) — FIM thắng magnitude mọi ratio. Còn thiếu: random-pruning baseline.
- Đo FLOPs / latency thực, không chỉ param count.
- ⚠️ Tất cả single seed (3407) — xác nhận thêm 1–2 seed trước khi chốt luận văn.

---

# APB QAT (Swin-S / Tiny ImageNet → CIFAR-100)

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

### APB **W + A** quant (thêm activation quant — LSQ, từ 2026-06-03)

Từ đây APB không còn weight-only: thêm `--act-bits` (LSQ signed, scope A = input của 96/100 APB Linear).
Activation là runtime → **không đổi eff-bit/packed** (vẫn tính trên weight); cột "A bit" là phần mới.

| #  | base | scope | br | DPLR λ | **A bit** | batch | ep | eff-bit (W) | **Best top-1** | packed | comp. |
| -- | ---- | ----- | -- | ------ | --------- | ----- | -- | ----------- | -------------- | ------ | ----- |
| C8     | (weight-only) | full | 0.99 | 3000 | — (FP32) | 64 | 30 | 1.77 | 88.55% | 10.8 MB | 22.7× |
| **C8+A8** | C8 | full | 0.99 | 3000 | **8** (LSQ) | 32 | 30 | 1.78 | **88.94%** | 10.8 MB | 22.5× |
| **C8+A2** | C8 | full | 0.99 | 3000 | **2** (LSQ) | 32 | 29 | 1.78 | **78.49%** | 10.8 MB | 22.5× |
| **C8+A1** | C8 | full | 0.99 | 3000 | **1** (Binary) | 32 | 29 | 1.78 | **51.03%** | 10.8 MB | 22.5× |

**Đường act-bit ↔ accuracy (full, br0.99, DPLR, batch32 — ablation SẠCH, chỉ khác act-bit):**
- **A8 88.94% → A2 78.49% (−10.45%) → A1 51.03% (−27.46% nữa).** Vách đá ở 1-bit: bước A2→A1 dốc gấp
  ~2.6× bước A8→A2. Hạ bit activation trên Swin **rất khắc nghiệt** (activation ViT 2 phía, nhiều outlier),
  và 1-bit (chỉ giữ dấu, bỏ magnitude) là cú sốc mạnh nhất. A1=51% vẫn >> random (1%).
- **1-bit dùng `BinaryActQuant` (sign·scale), KHÔNG phải LSQ** (LSQ degenerate ở 1-bit: Qp=0). Routing tự
  động qua `make_act_quant`. Commit `8f9b718`.
- **Plateau:** A1 đã bão hòa ~ep23-29 (50→51), khác A2 (chưa plateau ở ep29). 1-bit chạy thêm epoch lợi rất ít.
- **A8 ≈ free**: 88.94% còn nhỉnh hơn C8 weight-only 88.55%. ⚠️ **Confound batch:** C8 dùng batch 64,
  các A-run dùng batch 32 (OOM ở 64) → đừng quy +0.39% cho act-quant; chỉ kết luận "act 8-bit không tụt".
- Activation là runtime → eff-bit/packed/comp **không đổi** giữa A8/A2/A1/weight-only (10.8 MB, 22.5×). Giá trị
  của act-quant nằm ở **tốc độ/năng lượng inference + chuẩn W/A**, KHÔNG ở dung lượng file.

**⚠️ Tốc độ (ngược trực giác): 1-bit NHANH hơn 2-bit.** A1 Binary ~1154s/ep (~9.3h) < A2/A8 LSQ ~1450s/ep
(~11.7h) — nhanh ~20%. LSQ forward nhiều op/kernel hơn (div+clamp+round+mul+grad_scale), Binary chỉ 1 custom
Function. A1 tốc độ ổn định tuyệt đối (1152-1158s) → không memory thrashing.

Files: `2026-06-03_..._C8+A8_..._act8.md`, `2026-06-04_..._C8+A2_..._act2.md`, `2026-06-28_..._C8+A1_..._act1.md`.
TODO: A4 (lấp khoảng A8↔A2); chạy lại C8 batch32 (tách confound batch); sửa nhãn log `λ·dplr` (in dplr thô,
3 số không cộng khớp); 1-bit thử scope skip xem cứu được không; lặp cho scope skip (C6+A*).

### ⭐ Ablation PARTITION: FIMA-Q vs magnitude (flag `--partition`, từ 2026-06-28)

Tiêu chí chọn 1% weight giữ FP-outlier: `fim` (FIMA-Q, mặc định) | `magnitude` (APB gốc, giữ |w| lớn).

#### ⭐⭐⭐⭐⭐ BẢNG TỔNG HỢP — partition (magnitude vs fim) × DPLR loss × act-bit (8 ô, ĐẦY ĐỦ)

Chung: **full / br0.99 / 29ep / batch32 / seed3407**, init-baseline 90.88%. Mỗi ô là 1 run riêng (best top-1).

| act | partition | **no-DPLR** | **+DPLR (λ=3000)** | **DPLR giúp** | file (no-DPLR / DPLR) |
|:---:|:---------:|:-----------:|:------------------:|:-------------:|------------------------|
| **1** | fim       | 45.10% | 51.03% | **+5.93%** | `..._act1_fim_nodplr.md` / `..._act1.md` (C8+A1) |
| **1** | magnitude | 48.87% | **58.29%** | **+9.42%** | `..._act1_magnitude_nodplr.md` / `..._act1_magnitude.md` |
| **2** | fim       | 65.49% | 78.49% | **+13.00%** | `..._act2_fim_nodplr.md` / `2026-06-04_..._act2.md` (C8+A2) |
| **2** | magnitude | 77.98% | **81.79%** | **+3.81%** | `..._act2_magnitude_nodplr.md` / `2026-06-30_..._act2_magnitude.md` |

**Gap `magnitude − fim` (cùng act, cùng DPLR):**

| act | no-DPLR | +DPLR |
|:---:|:-------:|:-----:|
| **1** | +3.77% | +7.26% |
| **2** | +12.49% | +3.30% |

**3 kết luận từ bảng tổng hợp (8/8 ô):**
1. **magnitude > fim ở CẢ 8 ô** (mọi act-bit × mọi DPLR) → finding cực robust. FIM-cho-**partition** thua nhất quán.
2. **DPLR loss luôn dương** (+3.81 → +13.00 ở cả 4 cặp) → FIM-cho-**distillation** luôn giúp. ⇒ **giá trị FIMA-Q
   nằm ở distillation loss, KHÔNG ở partition.** Tốt nhất mỗi act = **magnitude+DPLR** (act1 58.29, act2 81.79).
3. **DPLR×partition đảo chiều theo act-bit:** act1 DPLR ưu ái magnitude (gap nới 3.77→7.26); act2 DPLR ưu ái fim
   (gap co 12.49→3.30, distillation bù cho fim-partition tệ + magnitude chạm trần).

⚠️ Tất cả single-seed (3407), br0.99/full/CIFAR-100. TODO: br0.95 (gap co?) + seed khác trước khi chốt thesis.

#### ⭐⭐⭐ br0.95 + act2 (init-baseline, full, 29ep, batch32, seed3407) — TRẢ LỜI TODO "gap co ở br0.95?"

3/4 ô của 2×2 (thiếu **magnitude+DPLR**). Init-baseline 90.88% → cũng lấp luôn "br0.95 full fair" còn thiếu.
Files: `..._br0.95_act2_fim_nodplr.md`, `..._br0.95_act2_magnitude_nodplr.md`, `..._br0.95_act2_fim_dplr.md`.

| partition \ DPLR | **no-DPLR** | **DPLR** | → DPLR giúp |
|------------------|-------------|----------|-------------|
| **fim** | 69.75% | 80.53% | **+10.78%** |
| **magnitude** | **82.63%** | _(chưa chạy)_ | — |
| **→ magnitude − fim** | **+12.88%** | — | |

**Kết luận br0.95/act2 (3 ô):**
1. **Gap magnitude−fim (no-DPLR) KHÔNG co lại:** +12.88% ở br0.95 vs +12.49% ở br0.99 → gần như đứng yên
   (nhích rộng chút). **BÁC BỎ** giả thuyết "fim lật lại / gap co khi bớt cực hạn br0.99→0.95". magnitude
   thắng fim **9/9 ô** đã đo (8 ô br0.99 + ô br0.95 no-DPLR).
2. **magnitude/no-DPLR (82.63) > fim/DPLR (80.53) = +2.10%:** magnitude **không** distillation vẫn hơn fim
   **có** distillation → củng cố mạnh "**giá trị FIMA-Q ở distillation-loss, KHÔNG ở partition**".
3. **DPLR vẫn giúp fim mạnh** (+10.78%, br0.99 là +13.00%) — distillation vá partition-fim tệ nhưng không đủ đuổi magnitude.
4. **br giảm 0.99→0.95 nâng đều mọi ô** (fair, cùng init-baseline): fim/no-DPLR +4.26 (65.49→69.75),
   magnitude/no-DPLR +4.65 (77.98→82.63), fim/DPLR +2.04 (78.49→80.53). Giữ nhiều FP hơn → acc cao hơn,
   nhưng **thứ hạng partition không đổi**.
5. eff-bit whole 4.29 / packed 26.2 MB / **9.3×** (br0.95 nén ít hơn br0.99 22.5× — đánh đổi acc↔size rõ).

**TODO còn lại:** (1) ô **magnitude+DPLR** br0.95/act2 để khép 2×2 (dự đoán ~83-84%, ceiling); (2) seed khác.

---

Ma trận 2×2 (partition × DPLR) ở **act-1 / full / br0.99 / 29ep / batch32 / seed3407**:

| partition \ DPLR | **no-DPLR** | **DPLR** | → DPLR giúp |
|------------------|-------------|----------|-------------|
| **fim** | **45.10%** | 51.03% | +5.93% |
| **magnitude** | **48.87%** | **58.29%** | +9.42% |
| **→ magnitude − fim** | **+3.77%** | **+7.26%** | |

Files: `..._act1.md` (fim+dplr), `..._act1_magnitude.md` (mag+dplr), `..._act1_magnitude_nodplr.md`, `..._act1_fim_nodplr.md`.

**Kết luận (2×2 ĐẦY ĐỦ, ở chế độ cực hạn act1/br0.99/full):**
1. **⚠️ magnitude > FIM ở CẢ HAI cột** (+3.77% no-DPLR, +7.26% DPLR) → finding **robust**, không nhờ DPLR. ĐI
   NGƯỢC giả thuyết "FIM partition tốt hơn".
2. **FIM-cho-partition đơn thuần TỆ HƠN magnitude** (45.10 < 48.87, no-DPLR sạch tuyệt đối): ở binarize cực hạn,
   FIM chọn outlier *hại* so với giữ |w| lớn.
3. **Tách vai trò FIM:** kém ở PARTITION nhưng tốt ở DISTILLATION — DPLR (loss dùng FIM) giúp cả hai (+5.93 fim,
   +9.42 mag). Giả thuyết: APB α·sign(w) sai số lớn nhất ở |w| lớn → giữ |w| lớn FP giảm trực tiếp sai số, lấn
   át curvature của FIM ở br0.99+act1.
4. **Tương tác super-additive:** gap magnitude−fim nới rộng khi bật DPLR (3.77 → 7.26); magnitude+DPLR tốt nhất (58.29).
5. **Tốc độ:** no-DPLR ~928s/ep (7.5h) nhanh ~18% vs DPLR ~1154s/ep (9.1h). magnitude+DPLR verify reproducible (2 lần đều 58.29%).

⚠️ **Đây là config CỰC HẠN NHẤT** — FIM có thể lật lại magnitude ở br0.95/act≥2. TODO: chạy 2×2 (hoặc fim-vs-mag)
ở **br0.95 + act2** TRƯỚC KHI kết luận; seed khác xác nhận. Nếu FIM thua mọi nơi → định vị lại đóng góp sang FIM-distillation.

**⭐⭐⭐ magnitude > fim ở CẢ act1 LẪN act2 (no-DPLR, cặp SẠCH — chỉ khác partition):**
| act | fim/no-DPLR | magnitude/no-DPLR | **mag − fim** |
|-----|-------------|-------------------|---------------|
| 1 | 45.10% | 48.87% | **+3.77%** |
| **2** | **65.49%** | **77.98%** | **+12.49%** |

→ **magnitude thắng fim ở mọi act-bit, gap NỚI RỘNG khi act tăng (3.77 → 12.49).** BÁC BỎ giả thuyết "fim lật
lại khi bớt cực hạn". act1 bị activation 1-bit che lấp partition (gap nhỏ); act2 activation tốt hơn → sai số tái
tạo weight chi phối → partition lộ rõ → gap lớn. Files `..._act2_magnitude_nodplr.md`, `..._act2_fim_nodplr.md`.

**⭐⭐⭐⭐ 2×2 ĐẦY ĐỦ partition × DPLR ở act2 (full / br0.99 / 29ep / batch32 / seed3407) — ĐÃ KHÉP (2026-06-30):**
| partition \ DPLR | **no-DPLR** | **DPLR** | → DPLR giúp |
|------------------|-------------|----------|-------------|
| **fim** | 65.49% | 78.49% (C8+A2) | **+13.00%** |
| **magnitude** | 77.98% | **81.79%** | +3.81% |
| **→ magnitude − fim** | **+12.49%** | **+3.30%** | |

Files: `..._act2_fim_nodplr.md`, C8+A2 (`2026-06-04_..._act2.md`), `..._act2_magnitude_nodplr.md`, `2026-06-30_..._act2_magnitude.md`.
1. **magnitude > fim ở CẢ 4 ô của act2** (và cả 4 ô act1) → 8/8 cell magnitude thắng. Finding cực robust.
2. **Tương tác DPLR×partition ĐẢO CHIỀU act1↔act2:** act1 DPLR giúp magnitude nhiều hơn → gap NỚI (3.77→7.26);
   act2 DPLR giúp **fim** nhiều hơn (+13.00 vs +3.81) → gap **CO** (12.49→3.30). Ở act2, distillation (FIM-loss)
   bù phần lớn cho partition tệ của fim; magnitude đã tốt nên ít chỗ cải thiện (ceiling). → **FIM giá trị ở
   DISTILLATION, không ở PARTITION** càng được củng cố: chính DPLR (dùng FIM) cứu fim, không phải fim-partition.
3. **magnitude+DPLR tốt nhất ở act2 (81.79%, quant cost −9.09%)** — đúng pattern act1 (mag+DPLR=58.29 cũng nhất).
   act-bit (mag+DPLR): act1 58.29 → act2 81.79 = **+23.50%** (hẹp hơn no-DPLR +29.11 vì DPLR nâng sàn act1).

**⚠️ HỆ QUẢ cho thesis (đối mặt thẳng):** FIM-cho-PARTITION **không giúp** (thua magnitude nhất quán) — kết quả
ÂM cho "FIMA-Q-guided partition". Nhưng FIM-cho-DISTILLATION (DPLR) VẪN giúp (+5.93/+9.42 ở act1). → **Định vị
lại: giá trị FIMA-Q nằm ở distillation loss, KHÔNG ở partition** (ít nhất br0.99/full). ✅ **UPDATE
(2026-06-30):** đã chạy br0.95/act2 (xem section "br0.95 + act2" phía trên) → gap **KHÔNG co** (+12.88% vs
+12.49%), kết luận giữ nguyên qua cả br. TODO còn: (1) khép ô magnitude+DPLR br0.95; (2) seed khác.
- (phụ) act2/magnitude/no-DPLR 77.98 ≈ C8+A2 fim+DPLR 78.49 (−0.51%) dù không distill; act-bit mag/no-DPLR: act1 48.87→act2 77.98 (+29.11%).

📌 **Đối chiếu paper APB (arXiv 2306.08960):** APB gốc weight-only, khi đo efficiency mới gắn activation
quant **2-bit** bằng quantizer riêng (EWGS). → Ta dùng LSQ (cùng họ learned-step) là đúng hướng paper;
2-bit cũng đúng mức paper dùng. Khác biệt = paper làm CNN, ta làm Swin (transformer) → phần novel.

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
