# Báo cáo đề tài NCKH — Nén Vision Transformer bằng Pruning + Quantization dẫn hướng bởi Fisher Information

> Tài liệu tiến độ. Model: `swin_small_patch4_window7_224` (Swin-S). Dataset chính: CIFAR-100.
> Toàn bộ số liệu lấy từ `runs/README.md` (log thực nghiệm). Mọi run single-seed 3407 — **chưa** đủ seed để chốt.
>
> ⛔ **DEPRECATED (2026-07-13):** `dplr`/`fim` cho **importance & partition** đã bỏ (bản tự chế, không rigorous
> để rank). Đại diện FIM cho ranking = **`fisher`** (chuẩn). **DPLR *loss* (distillation) giữ nguyên.** Hệ quả
> lên báo cáo: §6.1 bỏ hàng `dplr`; §6.4 (partition) — số `fim` cũ là `dplr` → **loại làm bằng chứng**, đang
> re-run bằng `fisher` (A2). Xem `ABLATIONS.md`.

---

## 1. Ý tưởng của đề tài

**Bài toán:** Vision Transformer (ViT/Swin) cho độ chính xác cao nhưng nặng (Swin-S ~48.9M tham số),
khó triển khai trên thiết bị hạn chế tài nguyên. Cần **nén mô hình** mà giữ được accuracy.

**Hai hướng nén được dùng chung một tín hiệu "độ quan trọng" (importance):**

1. **Structural pruning** — cắt bỏ hẳn các **attention head** và **MLP neuron** ít quan trọng nhất.
   → giảm số tham số + FLOPs thật, model nhỏ đi về mặt kiến trúc.
2. **APB quantization + QAT** — với mỗi trọng số, quyết định giữ **binary 1-bit** (`α·sign(w)`) hay
   **full-precision**; số ít trọng số "outlier" quan trọng được giữ FP để bảo toàn accuracy.

**Ý tưởng cốt lõi / điểm mới kỳ vọng:** thay vì dùng **magnitude** (|w|) như cách truyền thống để quyết định
cắt/giữ trọng số, ta dùng **importance dựa trên Fisher Information Matrix (FIM)** — lấy cảm hứng từ
**FIMA-Q** (DPLR-FIM, arXiv 2506.11543). FIM đo độ nhạy của loss theo từng tham số (bậc hai), về lý thuyết
"thông minh" hơn magnitude. **Đây là lần đầu áp tín hiệu FIM kiểu này cho Vision Transformer** ở cả 3 vai trò:
partition khi lượng tử hóa, chọn head/neuron khi pruning, và distillation loss khi fine-tune.

**⚠️ Thẳng thắn về kết quả (chi tiết ở §6–7):** giả thuyết "FIM tốt hơn magnitude" **chỉ đúng một phần**.
FIM **thắng** magnitude cho **pruning** và distillation-loss **luôn có ích**, nhưng FIM **thua** magnitude
khi dùng cho **partition lúc lượng tử hóa** (thua nhất quán 9/9 cấu hình đo được). Đây là một **kết quả âm**
có giá trị khoa học: nó buộc ta **định vị lại đóng góp** — giá trị của FIMA-Q nằm ở *distillation loss* và
*pruning importance*, **không** ở *partition*.

---

## 2. Cơ sở lý thuyết & các paper nền

| Paper | arXiv | Vai trò trong đề tài |
|---|---|---|
| **APB** — Neural Network Compression using Binarization and Few Full-Precision Weights (Nardini et al.) | 2306.08960 | Phương pháp lượng tử hóa lõi: `α·sign(w)` cho vùng binary, giữ số ít trọng số FP. |
| **FIMA-Q** — PTQ for ViT by Fisher Information Matrix Approximation (Wu, Wang et al.) | 2506.11543 | Nguồn của **importance**: DPLR-FIM (Diagonal + Low-Rank Fisher). |
| **EWC Done Right** — Logits Reversal (Liu & Chang) | 2603.18596 | Kỹ thuật LR chống gradient-vanishing khi ước lượng FIM. Đã thử rồi bỏ khỏi method (chốt fisher thường); giữ làm related-work. |
| **HEART-ViT** — Hessian-Guided Dynamic Attention & Token Pruning (Uddin et al.) | 2512.20120 | Related work: cũng dùng tín hiệu bậc-hai (Hessian) để prune head/token trên ViT. |

**Khái niệm importance (`compute_weight_dplr_fim`):** ước lượng từ `k` batch calibration bằng forward+backward:
- `diag` (L1): `E[|∇z|]`; `rank` (L2): `E[(∇z)²]`; `dplr = p1·rank + p2·diag`, rồi nhân `E[x²]` để bắc cầu sang trọng số.
- `fisher`: empirical Fisher chính xác theo token `F(W_ij) = (1/T)Σ_t g_i(t)²·x_j(t)²` (giữ tương quan g–x, không factorize).
- Tuỳ chọn `--importance-full` (chạy toàn dataset, hết sampling error). *(Logits Reversal đã thử rồi bỏ.)*

**⚠️ Lưu ý học thuật:** code là **"FIMA-Q-inspired"**, KHÔNG phải FIMA-Q nguyên bản (đã bỏ low-rank inverse thật,
AdaRound; dùng gradient của CE thay vì KL FP‖quant). Báo cáo nên ghi *"importance kiểu empirical-Fisher lấy cảm hứng
DPLR"*, không ghi *"dùng FIMA-Q để prune"*.

---

## 3. Pipeline tổng thể

```
[0] FP baseline:  finetune.py → Swin-S pretrained fine-tune CIFAR-100 → ckpt/best_swin.pth = 90.88%
                  (mốc để đo "cost" của mọi phép nén)

TRACK A — Structural pruning (prune_cifar.py)
  load baseline → importance (fisher) → prune head+MLP thấp nhất (global) → eval post-prune
  → finetune phục hồi → best_pruned_model.pt   [giảm tham số thật]

TRACK B — APB quantization + QAT (qat.py)
  load baseline/pruned → apply_apb (mask theo partition=magnitude) → eval PTQ
  → [opt] DPLR distillation loss → QAT (mask frozen, freeze α ở epoch/2)
  → best.pth + best_packed.pt (định dạng paper Eq 10)   [giảm bit/trọng số]

[C] KẾT HỢP:  TRACK A → TRACK B (qat.py --init-model best_pruned_model.pt)
```

Cấu hình đã chốt (2026-07-11): **prune dùng `--importance fisher`**, **QAT dùng `--partition magnitude`**
(đã đặt làm default trong code).

---

## 4. Chi tiết phương pháp

### 4.1 Structural pruning (`prune.py`, `prune_cifar.py`)
- Chỉ cắt **head** (WindowAttention: qkv rows, proj cols, relative_position_bias) và **MLP neuron**
  (fc1 rows, fc2 cols) — hai chỗ này **không đổi stage-dim** nên không lan shape sang block/downsample khác.
  Model sau prune là **timm Swin thuần**, chạy forward chuẩn (không cần wrapper).
- Importance mỗi head = Σ FIM trên qkv-rows + proj-cols của head đó; mỗi neuron = Σ FIM trên fc1-row + fc2-col.
- **`--rank-mode`**: `global` (gộp toàn cục, block "rẻ" bị cắt nhiều hơn — khuyến nghị) vs `per_layer` (cắt đều mỗi block).
  `--global-metric per_param` = xếp hạng theo importance/param-cost (ưu tiên nén). Sàn an toàn `--mlp-keep-frac 0.05`.

### 4.2 APB quantization (`qat.py`: `APBLinear`, `_APBSTE`)
- Mỗi `nn.Linear` mục tiêu → `APBLinear`: `mask=True` → `α·sign(w)` (binary), `mask=False` → `latent_weight` (FP32).
- **`_APBSTE`** (custom autograd) giữ gradient cho **CẢ** `latent_weight` **VÀ** `α` (STE ngây thơ làm mất grad `α`).
  `α` học được, init `mean|w|`, giữ dương qua `abs()`, freeze ở nửa số epoch. Weight-decay KHÔNG áp lên `α`/LSQ step.
- **`--binary-ratio`** (0.75/0.95/0.99) = % trọng số binarize. **Scope APB = luôn full** (100 Linear, gồm head + downsample; flag `--apb-scope` đã bỏ).
- **`--partition`** = tiêu chí chọn trọng số nào giữ FP: `magnitude` (giữ |w| lớn — default mới) / `fim`.
- **Mask FROZEN** suốt training để ổn định (đã verify: 0% mask-flip).

### 4.3 Activation quantization (W+A) — `--act-bits`
- `2..31` → **LSQ** (learned step, signed symmetric); `1` → **BinaryActQuant** (`scale·sign`, vì LSQ suy biến ở 1-bit).
- Là đại lượng **runtime** → KHÔNG đổi eff-bit/kích thước file; giá trị nằm ở tốc độ/năng lượng inference + chuẩn W/A.

### 4.4 DPLR distillation loss (`DPLRBlockLoss`) — `--use-dplr-loss --dplr-lambda`
- Loss per-block khớp output của student (đã lượng tử hóa) với FP teacher: `total = CE + λ·Σ_blocks L_DPLR`.
- `λ = 3000` (raw DPLR ~0.002 vs CE ~6 nên cần khuếch đại). Đây là chỗ FIMA-Q thực sự **có ích** (xem §7).

---

## 5. Thiết lập thí nghiệm
- **Model:** `swin_small_patch4_window7_224` (timm pretrained, 48.91M params, input 224×224).
- **Dataset:** CIFAR-100 (torchvision, 32→224 bicubic, chuẩn hóa ImageNet). *Trước đó dùng Tiny ImageNet, đã chuyển
  từ 2026-05-30.* CIFAR-100 là **test-bed để debug pipeline — KHÔNG so với Table 1 của paper** (task khác).
- **Chung:** AdamW (wd=1e-4 cho weight, 0 cho α), cosine LR, AMP fp16, seed 3407, batch 64 (32 khi bật act-quant do OOM).
- **Epoch (protocol chuẩn = 30):** finetune 20 (early-stop patience 5), prune-finetune 20, QAT **30**. ⚠️ Một số run
  đo trước đó (ablation partition §6.4 + kết quả chính combined §6.6) chạy **29 ep** — chênh 1 epoch, ảnh hưởng
  accuracy không đáng kể; **bản cuối sẽ re-run các config headline ở 30 ep** cho nhất quán với C7–C8 (đã là 30).
- **Môi trường:** `C:/Users/Admin/anaconda3/envs/fimaq_apb/python.exe` (torch/timm); GPU RTX 50xx (cu128).

---

## 6. Kết quả

> Chỉ giữ các run thuộc phương pháp đã chốt: **prune = fisher**, **QAT partition = magnitude + DPLR**,
> **init-baseline**, **CIFAR-100**. Các run cũ không liên quan (Tiny ImageNet, head-random C1–C4) đã lược bỏ.

**FP baseline CIFAR-100 = 90.88%** (48.91M params) — mốc so mọi "cost".

### 6.1 Structural pruning — chọn importance (global, per_param, mlp-keep-frac 0.05, 20 ep)
Ô = **best top-1 % (params M)** sau finetune.
| importance | ratio 0.5 | ratio 0.7 | ratio 0.9 |
|---|---|---|---|
| **fisher ⭐ (chốt)** | **90.39 (22.1)** | **87.50 (13.6)** | **71.07 (5.7)** |
| magnitude l2sq | 88.33 (25.8) | 79.01 (15.8) | 53.08 (5.9) |
| magnitude l1 | 87.60 (26.1) | 76.71 (16.0) | 51.31 (5.9) |
| ~~dplr~~ ⛔ deprecated | ~~90.87 (23.1)~~ | ~~87.80 (14.3)~~ | ~~63.00 (6.0)~~ |

→ **fisher (FIM chuẩn) THẮNG magnitude ở mọi ratio** (0.9 cách biệt lớn: 71.07% vs 53.08%) với ít tham số hơn
→ **chọn fisher** (bền nhất khi prune cực hạn). *(Hàng `dplr` — bản tự chế — đã deprecated, chỉ để tham khảo.)*

### 6.2 Structural pruning — chọn rank-mode (global vs per_layer)
| ratio | global (acc / params) | per_layer (acc / params) | → |
|---|---|---|---|
| 0.5 | **90.87% / 23.06M** | 90.55% / 25.32M | global thắng cả 2 |
| 0.7 | 87.80% / **14.27M** | 87.85% / 16.19M | acc ~hòa, global ít params hơn ~2M |
| 0.9 | **63.00% / 5.97M** | 60.64% / 6.20M | global thắng cả 2 |

→ **global ≥ per_layer** (cùng acc, ít params hơn) → **chọn global**. **Sweet spot 0.5–0.7**
(0.5 gần như free −0.01% @ −52.9%; 0.7 đổi −3% lấy −70.8%). **0.9 sụp đổ** (còn quá ít capacity).

### 6.3 APB-QAT — DPLR vs CE (init-baseline 90.88%, **scope full**, br0.99, 30 ep, seed 3407)
Scope APB chốt **luôn full** (100 Linear) → chỉ báo 2 run full C7/C8.
| # | DPLR λ | eff-bit (whole) | Post-APB | **Best top-1** | quant cost | packed | comp. |
|---|---|---|---|---|---|---|---|
| C7 | OFF (CE) | 1.77 | 1.54% | 87.44% | −3.44% | 10.8 MB | 22.7× |
| **C8 ⭐** | 3000 | 1.77 | 1.54% | **88.55%** | −2.33% | 10.8 MB | 22.7× |

→ **DPLR loss thắng CE-only (+1.11%)** → distillation là driver thật. C8 = điểm weight-only tốt nhất
(**88.55% @ 10.8 MB, 22.7×**). *(Bản cũ có scope=skip cho C6=89.32% acc cao hơn, nhưng flag `--apb-scope`
đã bỏ — xem §4.2; skip không còn trong phương pháp.)*

### 6.4 APB-QAT — chọn partition (**fisher** vs magnitude) — full/br0.99/act2/29 ep/batch32

> ✅ **Đã có bằng chứng fisher** (Kaggle 2026-07-03, `runs/2026-07-03_..._A2_fisher_partition_br0.99_act2.md`).
> Option đổi tên `fim` → `fisher` (bỏ tên mập mờ mặc-định-dplr). **magnitude thắng fisher**, và **fisher ≈ dplr**:

| partition | no-DPLR | +DPLR λ3000 |
|-----------|---------|-------------|
| **magnitude ⭐** | **77.98** | **81.79** |
| fisher | 65.78 | 77.80 |
| **gap (mag − fisher)** | **+12.20** | **+3.99** |

→ **magnitude > fisher cả 2 cột** (+12.20% no-DPLR, +3.99% +DPLR) → kết quả âm cho "FIM-guided partition" đã
chứng minh với **fisher**. **DPLR *loss* luôn dương** ⇒ **giá trị FIMA-Q ở distillation loss, KHÔNG ở
partition.** Config tốt nhất = **magnitude + DPLR**.

| act-bit | partition | no-DPLR | +DPLR |
|---|---|---|---|
| 1 | ~~fim (=dplr)~~ ⛔ | 45.10% | 51.03% |
| 1 | **magnitude ⭐** | 48.87% | **58.29%** |
| 2 | ~~fim (=dplr)~~ ⛔ | 65.49% | 78.49% |
| 2 | **magnitude ⭐** | 77.98% | **81.79%** |

→ (tạm) magnitude > dplr-partition mọi ô, nhưng **fisher-partition chưa test**. **DPLR *loss* luôn dương**
(+3.8→+13%) ⇒ **giá trị FIMA-Q ở distillation loss** (kết luận này không phụ thuộc partition, vẫn vững).
Config tốt nhất hiện tại mỗi mức = **magnitude + DPLR**.

### 6.5 APB-QAT — activation quant (W+A, xuất phát từ C8, full/br0.99/DPLR)
| cấu hình | A-bit | **Best top-1** |
|---|---|---|
| C8 (weight-only) | — | 88.55% |
| C8+A8 (LSQ) | 8 | **88.94%** (≈ free) |
| C8+A2 (LSQ) | 2 | 78.49% |
| C8+A1 (Binary) | 1 | 51.03% |

→ **Vách đá ở 1-bit** (bước A2→A1 dốc gấp ~2.6× A8→A2). Hạ bit activation trên ViT rất khắc nghiệt
(activation 2 phía, nhiều outlier). **A8 gần như miễn phí**; act là runtime → không đổi kích thước file.

### 6.6 ⭐ KẾT QUẢ CHÍNH — Prune + Quant KẾT HỢP (nén kép)
Đây là đóng góp cuối cùng của đề tài: lấy model **đã prune** (fisher, global, ratio 0.5 → **22.1M params,
−52.9%**) rồi **APB-QAT** trên đó (full / magnitude / DPLR λ=3000 / W+A / 29 ep). Init từ
`ckpt/best_pruned_g05_fisher.pt`. (Nguồn: 4 run wandb 2026-07-07.)

| br (weight) | act-bit | **Best top-1** | so với FP 90.88% | so với prune-only 90.39% |
|---|---|---|---|---|
| **0.95** | **2** | **83.25%** ⭐ | −7.63% | −7.14% |
| 0.99 | 2 | 81.19% | −9.69% | −9.20% |
| 0.95 | 1 | 62.20% | −28.68% | −28.19% |
| 0.99 | 1 | 52.85% | −38.03% | −37.54% |

**Nhận xét (điểm sáng để nhấn):**
- **Cấu hình tốt nhất: prune 0.5 + br0.95 + act2 = 83.25%** — model vừa **nhỏ 52.9% về kiến trúc** (22M
  thay vì 48.9M) **vừa** lượng tử hóa weight ~binary + activation 2-bit.
- **Pruning gần như "miễn phí" khi chồng lên quant:** combined **br0.99/act2 = 81.19%** so với **quant-only
  br0.99/act2 magnitude+DPLR = 81.79%** (§6.4) → thêm cả nhánh prune (−52.9% params) chỉ mất **−0.60%** accuracy.
  Đây là bằng chứng mạnh cho việc **kết hợp 2 hướng nén là cộng hưởng tốt**, không xung đột.
- **1-bit activation vẫn là điểm chết** (62%/53%) — nhất quán với §6.5; nén kép không cứu được vách đá 1-bit.

---

## 7. Nhận định trung thực (đối mặt thẳng với hạn chế)

**Kết quả chưa "đẹp" như kỳ vọng ban đầu — và đây là phần trung thực nhất của báo cáo:**

1. **Đóng góp "mới" nhất (FIM-guided partition) cho kết quả ÂM:** magnitude thắng FIM nhất quán ở partition.
   → Không thể bán câu chuyện "FIM partition tốt hơn". **Đã định vị lại**: đóng góp thực nằm ở
   (a) **FIM cho pruning** (thắng magnitude mọi ratio, §6.2), và (b) **DPLR distillation loss** (luôn +acc, §6.3–6.4).
2. **Nén cao ↔ mất accuracy rõ:** br0.99+full+act-quant thấp bit tụt mạnh (A1 chỉ 51%); pruning 0.9 sụp đổ.
   Vùng "vừa nén tốt vừa giữ acc" chỉ ở mức trung bình (prune 0.5–0.7; quant br0.99 full C8 88.55%).
3. **Lĩnh vực còn mới → ít baseline để so:** FIMA-Q (2025), HEART-ViT (2025) đều rất mới; việc áp FIM cho
   **structural pruning + APB partition trên Swin** gần như chưa có tiền lệ trực tiếp. Hệ quả: **khó đặt số của
   mình cạnh một bảng chuẩn** — hiện chỉ so *nội bộ* (FIM vs magnitude, DPLR vs CE, global vs per_layer) chứ
   chưa so được với SOTA ngoài. Đây vừa là **hạn chế** (thiếu điểm tựa) vừa là **khoảng trống** (novelty).
4. **Mọi số single-seed (3407):** chưa chạy nhiều seed → chưa khẳng định được ý nghĩa thống kê.
5. **Dataset là CIFAR-100 (test-bed), không phải ImageNet-1k** như paper → số accuracy chỉ dùng để **so tương đối**
   giữa các cấu hình, KHÔNG so trực tiếp bảng paper.

**Điểm tích cực để nhấn:** (i) pipeline hoàn chỉnh, tự động hóa, tái lập được (log đầy đủ ở `runs/`);
(ii) hai kết luận robust — *FIM có ích cho pruning* và *distillation loss luôn có ích*; (iii) một kết luận âm
rõ ràng, có bằng chứng (9/9 ô) — bản thân nó là đóng góp khoa học (biết chỗ nào FIM KHÔNG giúp);
(iv) **kết quả chính (nén kép, §6.6) cho thấy 2 hướng nén cộng hưởng tốt** — prune thêm −52.9% params chỉ
tốn −0.60% acc khi chồng lên quant (81.19% vs 81.79%), cấu hình tốt nhất **83.25% @ 22M params + W~1bit/A2bit**.

---

## 8. Tiến độ hiện tại

**✅ Đã xong:**
- FP baseline CIFAR-100 (90.88%).
- Structural pruning: sweep ratio 0.5/0.7/0.9 × {global, per_layer}; ablation importance {dplr, fisher, magnitude l1/l2sq}. *(fisher+LR+full từng chạy, đã bỏ khỏi method.)*
- APB-QAT (scope full): br {0.95, 0.99} × {CE, DPLR}; ma trận fair init-baseline C7–C8 (scope=skip C5/C6 cũ đã bỏ khỏi phương pháp).
- Activation quant W+A: A8/A2/A1 (LSQ + Binary).
- Ablation partition (magnitude vs fim) × DPLR × act-bit — 9/9 ô.
- **Pipeline prune→quant KẾT HỢP (kết quả chính, §6.6):** 4 run từ `best_pruned_g05_fisher.pt` (22M) →
  APB-QAT full/magnitude/DPLR/W+A, br{0.95,0.99}×act{1,2}. Best = **83.25%** (br0.95/act2). Đã có trên
  wandb (2026-07-07) nhưng **chưa** viết thành file `runs/*.md` — cần bổ sung log để hoàn tất tài liệu.
- Packed export theo Eq 10, logging + wandb backfill.

**🔜 Việc còn lại (TODO):**
- **Chuẩn hóa 30 ep:** re-run các config headline (§6.4 partition ablation + §6.6 combined) hiện ở 29 ep → 30 ep
  cho nhất quán toàn bộ (C7–C8 đã 30). Chênh nhỏ nhưng cần đồng bộ trước khi nộp.
- Chạy **nhiều seed** (≥2–3) để chốt ý nghĩa thống kê trước khi đưa vào luận văn.
- Prune ratio trung gian **0.6** (lấp khoảng 23M↔14M); thử 0.9 với 30–40 ep (kiểm tra capacity floor).
- Baseline **random-pruning** (để so với FIM/magnitude); đo **FLOPs/latency thực**, không chỉ param count.
- Khép ô còn thiếu: **magnitude+DPLR br0.95/act2**; thử act-bit 4 (lấp A8↔A2).
- Tách confound batch (C8 batch64 vs A-runs batch32) — chạy lại C8 batch32.
- (Nếu mở rộng) **unstructured pruning** dùng chung importance; thử trên ImageNet-1k để có số so paper.

---

## 9. Cấu trúc code (để reproduce)
- `apb_fimaq/qat.py` — pipeline APB-QAT (APBLinear, FIM extract, DPLR loss, act-quant, packed export).
- `apb_fimaq/prune.py` + `prune_cifar.py` — structural pruning + finetune.
- `apb_fimaq/finetune.py` — FP baseline. `apb_fimaq/fimaq_ptq_cifar.py` — PTQ baseline (dùng FIMA-Q/ gốc, chỉ import).
- `scripts/cifar_loader.py` — dataloader. `runs/*.md` + `runs/README.md` — log + bảng kết quả đầy đủ.

## 10. Tài liệu tham khảo
1. Nardini et al., *Neural Network Compression using Binarization and Few Full-Precision Weights (APB)*, arXiv:2306.08960.
2. Wu, Wang et al., *FIMA-Q: Post-Training Quantization for ViT by Fisher Information Matrix Approximation*, arXiv:2506.11543.
3. Liu & Chang, *Elastic Weight Consolidation Done Right for Continual Learning*, arXiv:2603.18596.
4. Uddin et al., *HEART-ViT: Hessian-Guided Efficient Dynamic Attention and Token Pruning in ViT*, arXiv:2512.20120.
5. Esser et al., *Learned Step Size Quantization (LSQ)*, ICLR 2020.
6. Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, arXiv:2103.14030.
