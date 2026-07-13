# ABLATIONS.md — Kế hoạch & trạng thái các ablation study (cho paper NCKH)

> Tài liệu theo dõi **từng ablation một**: mục tiêu, biến cô lập, control cần có, điều quan trọng nhất,
> và trạng thái. Nguồn số liệu: `runs/README.md` + `REPORT.md`. Cập nhật 2026-07-13.
>
> **Nguyên tắc xuyên suốt (3 điều):**
> 1. **Cô lập 1 biến** — mỗi cặp so sánh chỉ khác đúng 1 thứ; phần còn lại (seed / epoch / batch / init) giống hệt.
> 2. **Control đúng** — pruning cần `magnitude` (random đã bỏ, xem A1); QAT cần `PTQ-only`; distillation cần `CE-only`.
> 3. **So ở cùng "chi phí"** — pruning so cùng #params (Pareto); quant so cùng eff-bit; act so cùng batch.
>
> **⭐ QUYẾT ĐỊNH LỚN (2026-07-13) — bỏ `dplr` khỏi mọi vai trò IMPORTANCE/RANKING:**
> - `dplr` (p1·E[g²]+p2·E[|g|] ×E[x²]) là **biến thể tự chế** (FIMA-Q-inspired), KHÔNG rigorous đủ để xếp hạng.
> - **Đại diện FIM cho mọi ranking/quyết-định-cứng = `fisher`** (empirical Fisher chuẩn): dùng ở prune-importance (A1) VÀ partition (A2).
> - **DPLR *loss* (distillation, A3) GIỮ NGUYÊN** — dplr ổn khi làm mục tiêu mềm, chỉ không dùng để rank.
> - Hệ quả: mọi số `dplr`-importance / `dplr`-partition cũ → **loại khỏi bằng chứng chính** (có thể để appendix).

---

## Config CHỐT (mọi số headline phải về đây)

**QAT:** `--partition magnitude --use-dplr-loss --dplr-lambda 3000` + scope full + **`--epochs 30` + `--batch-size 32`** + seed 3407.
**Prune:** `--importance fisher --rank-mode global --global-metric per_param --mlp-keep-frac 0.05 --epochs 20`.

### ⚠️ 2 confound đang tồn tại trong data cũ
- **Epoch:** prune=20 (protocol riêng, OK) · một số QAT=29ep · C7/C8/A8=30ep. → lưới ablation nội bộ nhất quán, nhưng số headline phải chuẩn 30ep.
- **Batch:** weight-only C7/C8 = **batch 64**; mọi run có act-quant = **batch 32** (OOM ở 64). → đừng so chéo weight-only(64) vs act-quant(32).

**Hệ quả:** hiện KHÔNG có run nào nằm đúng config chốt (magnitude+DPLR+30ep+batch32) → cần **H1/H2/H3** (xem cuối file).

---

## Bảng tổng trạng thái

| # | Ablation | Model | Trạng thái | Còn thiếu |
|---|---|---|---|---|
| **A1** | Importance cho pruning (**fisher** vs magnitude) | gốc→prune | ✅ xong | ~~random~~, ~~dplr~~ (đều bỏ 2026-07-13) |
| **A2** | Partition (**fisher** vs magnitude) | gốc | ✅ **xong** (Kaggle 2026-07-03) — magnitude > fisher | — |
| **A3** | DPLR distillation (on/off) | gốc | ✅ xong | — |
| **B1** | rank-mode (global/per_layer) | gốc→prune | ✅ xong | — |
| **B2** | prune ratio sweep | gốc→prune | ✅ xong | (tùy chọn 0.6) |
| **C1** | binary-ratio (0.95/0.99) | gốc | ⚠️ có | chuẩn hóa 30ep cho headline |
| **C2** | activation bit (A8/A4/A2/A1) | gốc | ⚠️ có A8/A2/A1 | A4 (tùy chọn) |
| **D1** | prune / quant / combined | gốc+pruned | ✅ xong (§6.6) | chuẩn hóa 30ep (H1) |
| **D2** | PTQ-only vs QAT | gốc | ⚠️ ẩn | tách "Post-APB" thành hàng chính thức |
| **R1** | Multi-seed (≥3) | headline | ❌ thiếu | **bắt buộc trước khi chốt** |
| **R2** | FLOPs / latency thực | tất cả | ❌ thiếu | 1 script, 0 GPU train |
| **R3** | Kiến trúc khác (DeiT/ViT) | — | ❌ chưa số thật | mở rộng (tùy chọn) |

---

## Chi tiết từng ablation

### A1 — Importance cho pruning ✅
- **Chứng minh:** tín hiệu FIM (fisher) chọn head/neuron tốt hơn baseline magnitude.
- **Biến:** `--importance` ∈ {**fisher**, magnitude l1/l2sq}. Giữ cố định: ratio, rank-mode=global, finetune 20ep/lr, seed, mlp-keep-frac.
- **Quan trọng nhất:** so ở **cùng #params** (Pareto), KHÔNG so accuracy thẳng (params lệch 22.1↔23.1M).
- **Đã có (số đã đo):** fisher = 90.39/87.50/71.07 (ratio 0.5/0.7/0.9) vs magnitude l2sq = 88.33/79.01/53.08.
  **Kết luận: fisher > magnitude MỌI ratio** (0.9 cách biệt lớn: 71 vs 53). Vẫn vững sau khi bỏ dplr.
- **Quyết định 2026-07-13:** bỏ `random` (magnitude là baseline kinh điển Han et al.) **VÀ** bỏ cột `dplr`
  (biến thể tự chế, không dùng để rank). Kết luận gọn: *"fisher > magnitude"*.
- **Chạy thêm:** không (dữ liệu fisher + magnitude đã đủ; chỉ xóa cột dplr khi trình bày).

### A2 — Partition: **fisher** vs magnitude ✅ XONG (magnitude thắng)
- **Chứng minh:** FIM (fisher) cho *partition* có giúp hơn magnitude không → **KHÔNG.**
- **Biến:** `--partition fisher` vs `--partition magnitude`. Cố định: br0.99/act2/29ep/batch32/seed3407.
- **⭐ Đổi tên 2026-07-13:** option cũ `fim` (mặc định dplr) → gọi thẳng `fisher`. `--partition {fisher, magnitude}`.
- **✅ Kết quả (Kaggle 2026-07-03, file `runs/2026-07-03_..._A2_fisher_partition_br0.99_act2.md`):**

  | partition | no-DPLR | +DPLR λ3000 |
  |-----------|---------|-------------|
  | **magnitude ⭐** | **77.98** | **81.79** |
  | **fisher** | 65.78 | 77.80 |
  | **gap (mag − fisher)** | **+12.20** | **+3.99** |

  → **magnitude > fisher cả 2 cột** (+12.20% no-DPLR, +3.99% +DPLR).
- **Kết luận:** magnitude > FIM-partition **robust qua cả fisher (rigorous) lẫn dplr** → **kết quả âm cho
  "FIM-guided partition" ĐÃ CHỨNG MINH.** Giá trị FIMA-Q ở **distillation loss (A3)**, không ở partition. **Story an toàn, không lật.**
- **Chạy thêm:** không (chỉ đo ở br0.99/act2 — đủ; muốn chắc hơn có thể thêm br0.95/act2 nhưng không bắt buộc).

### A3 — DPLR distillation: on vs off ✅
- **Chứng minh:** distillation loss dùng FIM (per-block) giúp QAT.
- **Biến:** `--use-dplr-loss` on/off. Giữ cố định: **cùng partition**, br, act, epoch, batch.
- **Quan trọng nhất:** so ở **cùng partition** + **ghi rõ λ=3000** (λ=1 mặc định là vô hiệu vì raw DPLR ~0.002 << CE ~6).
- **Đã có:** §6.3 (C7 CE vs C8 DPLR, +1.11%) + mọi cặp trong lưới §6.4 (+3.8→+13%). Kết luận: **DPLR luôn dương** ⇒ *giá trị FIMA-Q ở distillation loss, không ở partition.*
- **Chạy thêm:** không.

### B1 — rank-mode: global vs per_layer ✅
- **Quan trọng nhất:** report **CẢ accuracy LẪN #params** (global thắng vì ít params ở cùng acc).
- **Đã có:** ratio {0.5,0.7,0.9}. Kết luận: **global ≥ per_layer** mọi mức.

### B2 — prune ratio sweep ✅
- **Quan trọng nhất:** trình bày **Pareto** + đánh dấu sweet spot (0.5–0.7).
- **Đã có:** 0.5 / 0.7 / 0.9. (Tùy chọn: thêm 0.6 lấp khoảng 23M↔14M.)

### C1 — binary-ratio: 0.95 vs 0.99 ⚠️
- **Quan trọng nhất:** **cùng init** (đừng so br0.95 head-random vs br0.99 init-baseline — lỗi C1–C4 cũ). Cùng scope, partition.
- **Đã có:** một phần. **Cần:** số headline chuẩn hóa 30ep (nằm trong H2/H3).

### C2 — activation bit: A8/A4/A2/A1 ⚠️
- **Quan trọng nhất:** **CÙNG batch size.** Đừng so A8(batch32) với C8 weight-only(batch64). Các A-run đều batch32 → so *nội bộ* OK.
- **Đã có:** A8 / A2 / A1 (vách đá 1-bit). **Thiếu:** A4 (lấp khoảng A8↔A2) — tùy chọn.

### D1 — prune-only / quant-only / combined ✅
- **Quan trọng nhất:** so **combined vs quant-only ở CÙNG br/act** → ra thông điệp "prune thêm −52.9% params chỉ mất −0.6% acc" (81.19 vs 81.79).
- **Đã có:** §6.6 (4 run combined). **Cần:** chuẩn hóa 30ep + viết file log (H1).

### D2 — PTQ-only vs QAT ⚠️
- **Quan trọng nhất:** cùng model, chỉ khác có train QAT hay không.
- **Đã có (ẩn):** số "Post-APB (~1.5%)" chính là PTQ-only. **Cần:** tách thành hàng chính thức cạnh số sau-QAT.

### R1 — Multi-seed ❌ (BẮT BUỘC)
- **Quan trọng nhất:** report **mean ± std trên ≥3 seed** cho config headline (không cần cả lưới). Nếu std > khoảng chênh thì kết luận sụp.

### R2 — FLOPs / latency ❌
- **Quan trọng nhất:** đo **cùng batch/hardware/input-size** cho mọi model. 1 script, không train lại.

### R3 — DeiT/ViT ❌ (mở rộng)
- Code đã tổng quát hóa nhưng chưa có số accuracy thật; chạy 1 config sẽ nâng novelty "áp FIM cho họ ViT" từ claim → bằng chứng.

---

## Rerun headline cần làm (H1/H2/H3)

Kéo mọi số headline về **1 config thống nhất: magnitude + DPLR + 30ep + batch32**. Dọn sạch cả confound epoch **lẫn** batch.

| | Là gì | Thay số cũ lệch | Model / init |
|---|---|---|---|
| **H1** | Best **combined** (nén kép) | §6.6 best 83.25% (29ep) | pruned: `ckpt/best_pruned_g05_fisher.pt` ✅ có |
| **H2** | Best **quant-only W+A** | lấp ô mag+DPLR br0.95/act2 + 30ep | gốc: `ckpt/best_swin.pth` ✅ (default qat.py) |
| **H3** | Best **quant-only weight-only** | §6.3 C8=88.55% (đang **fim**, batch64) | gốc: `ckpt/best_swin.pth` ✅ (default qat.py) |

```bash
PY=C:/Users/Admin/anaconda3/envs/fimaq_apb/python.exe
# H1 — combined (pruned model)
$PY apb_fimaq/qat.py --dataset cifar100 --init-model ckpt/best_pruned_g05_fisher.pt \
  --binary-ratio 0.95 --act-bits 2 --partition magnitude \
  --use-dplr-loss --dplr-lambda 3000 --epochs 30 --batch-size 32 --seed 3407 \
  --out-dir ckpt/H1_combined_br095_act2_mag_30ep
# H2 — quant-only W+A (model gốc)
$PY apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.95 --act-bits 2 --partition magnitude \
  --use-dplr-loss --dplr-lambda 3000 --epochs 30 --batch-size 32 --seed 3407 \
  --out-dir ckpt/H2_br095_act2_mag_30ep
# H3 — quant-only weight-only (model gốc)
$PY apb_fimaq/qat.py --dataset cifar100 --binary-ratio 0.99 --partition magnitude \
  --use-dplr-loss --dplr-lambda 3000 --epochs 30 --batch-size 32 --seed 3407 \
  --out-dir ckpt/H3_C8_weightonly_mag_30ep
```

**✅ Baseline gốc:** `ckpt/best_swin.pth` (verify 2026-07-13: state_dict Swin-S đầy đủ, `head.fc.weight=(100,768)`
→ CIFAR-100, 90.88%). qat.py **mặc định init từ file này** (không cần `--baseline-ckpt`). ⇒ **H2/H3 + A2-fisher
KHÔNG bị chặn** — chạy được ngay. (`best.pth` cũ không cần nữa; best_swin.pth thay thế.)

---

## Nhật ký quyết định
- **2026-07-13:** ⭐⭐ **Đổi tên partition `fim` → `fisher`, GIỮ A2.** `--partition {fisher,magnitude}`.
  ✅ **A2 đã có bằng chứng fisher** — hóa ra đã chạy trên **Kaggle 2026-07-03** (grep local không thấy vì Kaggle
  ephemeral); đã chép về `runs/2026-07-03_..._A2_fisher_partition_br0.99_act2.md`. **magnitude > fisher (+12.20%
  no-DPLR, +3.99% +DPLR); fisher ≈ dplr.** Kết quả âm robust → story an toàn. A2 XONG, không cần chạy thêm.
  Cũng chép về 4 run combined (§6.6) `runs/2026-07-06_..._combined_prune_quant.md`.
- **2026-07-13:** ⭐ **bỏ `dplr` khỏi mọi vai trò importance/ranking** — dplr là biến thể tự chế, không rigorous.
  Đại diện FIM cho ranking = **fisher** (chuẩn), chỉ dùng ở pruning (A1). A1 vẫn vững (fisher > magnitude).
- **2026-07-13:** bỏ `random`-pruning baseline khỏi A1 (chấp nhận magnitude làm control kinh điển).
- **2026-07-13:** chốt rerun H1/H2/H3 về magnitude+DPLR+30ep+batch32.
- **2026-07-11:** chốt prune=`fisher`, QAT partition=`magnitude` (giá trị FIMA-Q ở distillation, không ở partition).
- Quy ước: ablation QAT chạy trên **model gốc** (cô lập biến); **pruned chỉ dùng cho combined** (§6.6/H1). Phù hợp convention literature (Deep Compression: ablate từng stage trên base, pipeline báo cáo tích lũy).
