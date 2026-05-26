# APB + FIMA-Q trên Swin Transformer — Hướng Dẫn Triển Khai

> **Mục tiêu:** Compress Swin Transformer bằng cách kết hợp FIMA-Q (Hessian-based PTQ quantization) + APB (binarization + pruning), dùng FIM-based importance thay cho magnitude threshold của APB gốc.  
> **Baseline repo:** https://github.com/ShiheWang/FIMA-Q  
> **Framework:** PyTorch

---

## Mục Lục

1. [Tổng quan 2 Papers](#1-tổng-quan-2-papers)
2. [Kiến trúc Swin Transformer](#2-kiến-trúc-swin-transformer)
3. [FIMA-Q — Chi tiết](#3-fima-q--chi-tiết)
4. [APB — Chi tiết](#4-apb--chi-tiết)
5. [Kết hợp APB + FIMA-Q](#5-kết-hợp-apb--fima-q)
6. [Pipeline triển khai](#6-pipeline-triển-khai)
7. [Các quyết định thiết kế cần đưa ra](#7-các-quyết-định-thiết-kế-cần-đưa-ra)
8. [Thứ tự implement từng bước](#8-thứ-tự-implement-từng-bước)

---

## 1. Tổng quan 2 Papers

### Paper 1: FIMA-Q (arXiv:2506.11543v1)

- **Loại:** Post-Training Quantization (PTQ) cho ViT
- **Core idea:** Dùng Fisher Information Matrix (FIM) để đo sensitivity của block output với quantization error, thay vì squared gradient như BRECQ
- **Key insight:**

$$\nabla \mathcal{L}_{KL}(\Delta z^{(b)}) = \mathbf{F}(z^{(b)}) \Delta z^{(b)}$$

→ FIM **tỉ lệ tuyến tính** với gradient KL divergence (không phải bình phương như BRECQ)

- **Approximation:** DPLR-FIM = Diagonal + Low-Rank FIM
- **Loss cuối:** `L_DPLR = α·L_rank-k + (1-α)·L_diag`
- **Setting:** Block-wise reconstruction, rank k=15, calibration 1024 images
- **Kết quả trên Swin:**

| Model  | Bits | Full-prec | FIMA-Q | vs QDrop* |
|--------|------|-----------|--------|-----------|
| Swin-S | 3/3  | 83.23     | 77.26  | +2.59%    |
| Swin-B | 3/3  | 85.27     | 78.82  | +2.25%    |
| Swin-S | 4/4  | 83.23     | 81.82  | +0.61%    |
| Swin-B | 4/4  | 85.27     | 83.60  | +0.61%    |

---

### Paper 2: APB (arXiv:2306.08960v2)

- **Loại:** QAT compression = Binarization + Pruning kết hợp
- **Core idea:** Mỗi weight hoặc binary `{-α, +α}` hoặc full-precision dựa trên magnitude threshold
- **Operator:**

$$\text{APB}(w) = \begin{cases} \text{sign}(w) \cdot \alpha & \text{nếu } |w| \leq \alpha + \delta \\ w & \text{ngược lại} \end{cases}$$

- **α và δ:** Learnable parameters, per layer
- **Memory:** `W = W_binary + W_sparse_fullprec`
- **Chưa apply cho ViT/Swin** — chỉ test trên CNN (ResNet, VGG)
- **Improvement của bạn:** Thay magnitude threshold bằng **FIM-based importance** từ FIMA-Q

---

## 2. Kiến trúc Swin Transformer

### 2.1 Tổng quan 4 stages (Swin-S, C=96)

```
Input Image (H×W×3)
        │
   Patch Partition + Linear Embedding   ← KHÔNG compress
        │
   Stage 1: H/4 × W/4 × C     (2 Swin Blocks)
        │ Patch Merging                 ← KHÔNG compress
   Stage 2: H/8 × W/8 × 2C    (2 Swin Blocks)
        │ Patch Merging                 ← KHÔNG compress
   Stage 3: H/16 × W/16 × 4C  (6 Swin Blocks)
        │ Patch Merging                 ← KHÔNG compress
   Stage 4: H/32 × W/32 × 8C  (2 Swin Blocks)
        │
   Classification Head                  ← KHÔNG compress
```

### 2.2 Bên trong một Swin Block

```
Input x (B, num_windows, window_size², C)
  │
  ├─ LayerNorm 1                         ← KHÔNG compress
  │
  ├─ W-MSA hoặc SW-MSA:
  │    ├─ qkv: Linear(C, 3C)            ← APB candidate (cẩn thận)
  │    ├─ proj: Linear(C, C)            ← APB candidate
  │    └─ relative_position_bias_table  ← KHÔNG compress (nhỏ, quan trọng)
  │
  ├─ LayerNorm 2                         ← KHÔNG compress
  │
  └─ MLP (FFN):
       ├─ fc1: Linear(C, 4C)            ← APB candidate (ưu tiên)
       └─ fc2: Linear(4C, C)            ← APB candidate (ưu tiên)
```

### 2.3 Weight shapes qua các stages (Swin-S, C=96)

| Stage | Layer | Shape          | Params  |
|-------|-------|----------------|---------|
| 1     | qkv   | [288, 96]      | 27,648  |
| 1     | proj  | [96, 96]       | 9,216   |
| 1     | fc1   | [384, 96]      | 36,864  |
| 1     | fc2   | [96, 384]      | 36,864  |
| 2     | qkv   | [576, 192]     | 110,592 |
| 2     | proj  | [192, 192]     | 36,864  |
| 2     | fc1   | [768, 192]     | 147,456 |
| 2     | fc2   | [192, 768]     | 147,456 |
| 3     | qkv   | [1152, 384]    | 442,368 |
| 3     | proj  | [384, 384]     | 147,456 |
| 3     | fc1   | [1536, 384]    | 589,824 |
| 3     | fc2   | [384, 1536]    | 589,824 |
| 4     | qkv   | [2304, 768]    | 1,769,472 |
| 4     | proj  | [768, 768]     | 589,824 |
| 4     | fc1   | [3072, 768]    | 2,359,296 |
| 4     | fc2   | [768, 3072]    | 2,359,296 |

> **Lưu ý:** FFN (fc1 + fc2) chiếm ~2/3 tổng params → ưu tiên compress FFN trước

---

## 3. FIMA-Q — Chi tiết

### 3.1 Vấn đề của BRECQ (baseline cũ)

BRECQ xấp xỉ diagonal FIM bằng squared gradient:

$$-\mathbf{H}(z^{(b)}) \approx \text{Diag}\left(\left(\frac{\partial \mathcal{L}}{\partial z^{(b)}_1}\right)^2, \ldots, \left(\frac{\partial \mathcal{L}}{\partial z^{(b)}_a}\right)^2\right)$$

**Sai vì:** bỏ qua variance term trong FIM:

$$\text{Diag}\,\mathbf{F}(z^{(b)}) = \mathbb{E}\left[\left(\nabla \log p\right)^2\right] = \underbrace{\mathbb{E}\left[\nabla \log p\right]^2}_{\approx \text{BRECQ}} + \underbrace{\text{Var}(\nabla \log p)}_{\text{bị bỏ qua}}$$

### 3.2 Key theorem của FIMA-Q

**Theorem 3.2:**

$$\mathcal{L}_{KL}(\Delta z^{(b)}) = \frac{1}{2} \Delta z^{(b)\top} \mathbf{F}(z^{(b)}) \Delta z^{(b)}$$

Differentiate cả 2 vế:

$$\nabla \mathcal{L}_{KL}(\Delta z^{(b)}) = \mathbf{F}(z^{(b)}) \Delta z^{(b)}$$

→ Diagonal FIM đúng:

$$\mathbf{F}_{diag}(z^{(b)}) = \text{Diag}\left(\frac{\nabla \mathcal{L}_{KL}(\Delta z^{(b)})_1}{\Delta z^{(b)}_1}, \ldots, \frac{\nabla \mathcal{L}_{KL}(\Delta z^{(b)})_a}{\Delta z^{(b)}_a}\right)$$

### 3.3 Bốn dạng loss

**1. Diagonal:**
$$\mathcal{L}_{diag} = \left(\frac{\nabla \mathcal{L}_{KL}}{\Delta z^{(b)}}\right)^\top \left(\Delta z^{(b,i)}\right)^2$$

**2. Rank-one:**
$$\mathcal{L}_{rank-1} = \frac{\left(\Delta z^{(b,i)\top} \nabla \mathcal{L}_{KL}\right)^2}{\nabla \mathcal{L}_{KL}^\top \Delta z^{(b,i)}}$$

**3. Low-rank (rank-k):**
$$\mathcal{L}_{rank-k} = \underbrace{\Delta z^{(b,i)\top} \nabla \mathcal{L}_{KL}}_{A} \cdot \underbrace{\left(\Delta z^{(b)\top} \Delta z^{(b)}\right)^{-1}}_{B} \cdot \underbrace{\Delta z^{(b)\top} \Delta z^{(b,i)}}_{C}$$

**4. DPLR (dùng cái này):**
$$\mathcal{L}_{DPLR} = \alpha \cdot \mathcal{L}_{rank-k} + (1-\alpha) \cdot \mathcal{L}_{diag}$$

### 3.4 Block-wise pipeline của FIMA-Q

```
Với mỗi Swin Block b:

Step 1: Forward để lấy outputs
    z_full  = Block_full(x_raw)       ← full-precision output
    z_quant = Block_quant(x_quant)    ← quantized output
    Δz(b)   = z_quant - z_full

Step 2: Forward qua phần còn lại
    O_full  = RestOfNetwork(z_full)
    O_quant = RestOfNetwork(z_quant)
    L_KL    = KL(O_quant || O_full)

Step 3: Backward để lấy gradient
    ∇L_KL(Δz(b)) = autograd.grad(L_KL, Δz(b))

Step 4: Progressive rank update (mỗi x iterations, tăng rank lên 1)
    k: 1 → 2 → ... → 15
    Tích lũy Δz(b) và ∇L_KL vào list

Step 5: Compute DPLR loss và optimize
    L_DPLR = α·L_rank-k + (1-α)·L_diag
    Update AdaRound weights + activation scaling factors
```

### 3.5 Hyperparameters của FIMA-Q

| Param | Giá trị | Ghi chú |
|-------|---------|---------|
| rank k | 15 | Tăng dần từ 1 |
| interval x | mỗi 500 iter tăng rank | |
| calibration | 1024 images (ImageNet) | |
| max_iter | 10000 | |
| α (loss weight) | 0.5 | blend diag + LR |
| quantizer | uniform, channel-wise (W), layer-wise (A) | |

---

## 4. APB — Chi tiết

### 4.1 Motivation

**Observation quan trọng:** Large absolute-value weights đóng vai trò **ngược nhau** trong 2 kỹ thuật:

| Kỹ thuật | Large |w| weight | Small |w| weight |
|----------|--------|----------|
| Binarization | Khó flip sign → **dead weight**, cản optimization | Dễ train |
| Pruning | **Quan trọng nhất** → phải giữ | Loại bỏ |

→ APB: giữ large |w| ở **full-precision** (như pruning muốn), binarize small |w| (tránh dead weight của binarization)

### 4.2 APB Operator — toán học đầy đủ

**Chuẩn hóa:**
$$\hat{w} = \frac{|w| - \alpha}{\delta}$$

**Operator:**
$$\text{APB}(w) = \begin{cases} \text{sign}(w) \cdot \alpha & \text{nếu } \hat{w} \leq 1 \text{ (binary zone)} \\ w & \text{nếu } \hat{w} > 1 \text{ (FP zone)} \end{cases}$$

**Indicator function:** $\chi_B = \mathbf{1}(\hat{w} \leq 1)$

### 4.3 Gradient flow — Straight Through Estimator

APB không differentiable tại boundary → dùng STE với $g(w) = w$:

$$\frac{\partial \text{APB}}{\partial w} = 1 \quad \text{(cả binary và FP zone)}$$

→ Gradient **pass through không đổi** qua APB operator.

**Gradient qua $\hat{w}$:**
$$\frac{\partial \mathcal{L}}{\partial \hat{w}} = \frac{\partial \mathcal{L}}{\partial w} \cdot \delta \cdot \text{sign}(w)$$

**Gradient của $\alpha$** (chỉ từ binary weights):
$$\frac{\partial \mathcal{L}}{\partial \alpha} = -\frac{1}{\delta n} \sum \frac{\partial \mathcal{L}}{\partial \hat{w}_i} \cdot \chi_B$$

**Gradient của $\delta$** (chỉ từ binary weights):
$$\frac{\partial \mathcal{L}}{\partial \delta} = \frac{1}{\delta^2 n} \sum \frac{\partial \mathcal{L}}{\partial \hat{w}_i} \cdot (\alpha - |w_i|) \cdot \chi_B$$

> **Critical:** Chỉ binary weights ($\chi_B = 1$) đóng góp vào gradient của $\alpha$ và $\delta$. Full-precision weights không ảnh hưởng.

### 4.4 Initialization

```
Với mỗi layer i (sau load pretrained):
    α_i = mean(|w_i|)      ← mean của absolute values
    δ_i = 3 * std(w_i)     ← 3 sigma
```

Giả sử $w \sim \mathcal{N}(\mu, \sigma^2)$ → interval này bao phủ ~99.7% weights ban đầu → **high compression ngay từ đầu**.

### 4.5 Training schedule

```
Epoch 0 → T/2:
    Update: W_latent, α, δ
    Binary/FP partition thay đổi liên tục

Epoch T/2 → T:
    FREEZE α và δ  ← partition cố định
    Chỉ update: W_latent (fine-tune surviving FP weights)
```

### 4.6 Weight decay — điều chỉnh compression rate

```
L_total = L_task + λ * ||W||²

→ W_latent bị kéo về 0 dần
→ |w| giảm → nhiều weight rơi vào [0, α+δ]
→ Binary ratio tăng

Điều chỉnh: λ tăng → compress hơn, λ giảm → accuracy cao hơn
NOTE: KHÔNG apply weight decay lên α và δ
```

### 4.7 Memory layout — cách lưu trữ

**Decomposition:**
$$W = W^{bin} + W^{full}$$

```
W_bin[i]  = α * sign(w[i])   ← TẤT CẢ entries, kể cả FP zone
W_full[i] = w[i] - W_bin[i]  ← nếu i trong FP zone
           = 0                ← nếu i trong binary zone
```

**Storage sau training:**
```
Lưu: α (1 scalar per layer)
     W_bin (bitpacked, 1 bit/weight, full matrix)
     W_sparse = (fp_indices, fp_values)   ← chỉ FP entries

Memory = n * 1bit + s * (32 + b_p) bits
b_p = ceil(log2(layer_size - 1)) + 1   ← bits cho index
s = số FP entries (mục tiêu: < 5% of n)
```

**Inference:**
```
C = A * B
  = (A_bin + A_full) * B
  = binary_MM(A_bin, B) + sparse_dense_MM(A_full, B)
```

---

## 5. Kết hợp APB + FIMA-Q

### 5.1 Ý tưởng core

**APB gốc:** dùng `|w| ≤ α+δ` để quyết định binary/FP → không tối ưu vì weight nhỏ chưa chắc ít quan trọng với task loss.

**Cải tiến:** dùng **FIM importance** từ FIMA-Q:

```
FIM_importance(w_ij) thấp  → weight ít ảnh hưởng đến output → binarize an toàn
FIM_importance(w_ij) cao   → weight quan trọng → giữ full-precision
```

### 5.2 Bridge: propagate FIM từ block output xuống weight

FIMA-Q tính FIM theo **block output** $z^{(b)}$, APB cần theo **weight** $w_{ij}$.

**Propagation qua chain rule:**

$$F_{diag}(w_{ij}) \approx \sum_k F_{diag}(z^{(b)}_k) \cdot \left(\frac{\partial z^{(b)}_k}{\partial w_{ij}}\right)^2$$

**Thực tế:** $\left(\frac{\partial z^{(b)}_k}{\partial w_{ij}}\right)^2$ là **squared gradient** của weight → có sẵn từ backward pass của FIMA-Q, không cần tính thêm.

### 5.3 So sánh APB gốc vs APB+FIMA-Q

| Thành phần | APB gốc | APB + FIMA-Q |
|-----------|---------|-------------|
| Partition decision | `|w| ≤ α+δ` (magnitude) | `F_diag(w) ≤ τ` (FIM importance) |
| Loss function | Cross-entropy (full dataset) | DPLR-FIM (1024 calib images) |
| Training mode | QAT | PTQ |
| α, δ | Learnable via backprop | Có thể fixed hoặc learnable |
| Architecture | CNN only | ViT/Swin |
| Gradient của α,δ | Từ task loss | Từ DPLR-FIM loss (formula giữ nguyên) |

### 5.4 Vấn đề đặc thù với Swin

**QKV projection:**
```python
qkv = Linear(C, 3*C)   # ghép Q, K, V thành 1 layer

# Weight shape: [3C, C]
# Thực ra: [Q_w | K_w | V_w], mỗi phần [C, C]

# Vấn đề: Q, K ảnh hưởng attention pattern (rất nhạy)
#          V ảnh hưởng value aggregation (ít nhạy hơn)
# → FIM của 3 phần khác nhau

# Giải pháp: tách riêng hoặc dùng per-head threshold
```

**Window attention:**
```
Block output shape: (B, num_windows, window_size², C)
# Khác plain ViT: (B, seq_len, C)
# FIMA-Q vẫn handle được nhưng cần reshape đúng
# FIMA-Q repo đã xử lý cho Swin → giữ nguyên
```

**Patch Merging:**
```
KHÔNG compress — đây là downsampling layer
Theo APB convention: giữ full-precision
```

---

## 6. Pipeline triển khai

### 6.1 Pipeline tổng quan

```
┌─────────────────────────────────────────────┐
│           PHASE 1: FIMA-Q Quantization       │
│                                             │
│  Swin Pretrained                            │
│       ↓                                     │
│  Per Swin Block:                            │
│    1. Compute Δz(b), ∇L_KL(Δz(b))          │
│    2. Build DPLR-FIM (rank-15)              │
│    3. Optimize AdaRound + activation scale  │
│    4. [NEW] Extract F_diag(w) per weight    │
│       ↓                                     │
│  → Quantized Swin + FIM importance scores  │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│           PHASE 2: APB Binarization         │
│                                             │
│  Per Linear layer (FFN trước, MHA sau):    │
│    1. Threshold τ từ F_diag(w)              │
│    2. mask_binary = F_diag(w) < τ           │
│    3. Apply APB operator                    │
│    4. Reconstruct với DPLR-FIM loss         │
│       ↓                                     │
│  → Binary + sparse FP weight format        │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│           PHASE 3: Storage                  │
│  Per layer: α + W_bin (bitpacked)           │
│             + W_sparse (indices + values)   │
└─────────────────────────────────────────────┘
```

### 6.2 Layers bị compress trong Swin

```
Layer                          Compress?    Ghi chú
─────────────────────────────────────────────────────
Patch Partition + Embedding    ✗            Layer đầu
LayerNorm                      ✗            < 0.5% params
relative_position_bias_table   ✗            Nhỏ, critical
Patch Merging (reduction)      ✗            Downsampling
W-MSA / SW-MSA: qkv            ⚠ Cẩn thận  Q,K nhạy cảm — làm sau
W-MSA / SW-MSA: proj           ✓            APB candidate
MLP: fc1                       ✓✓           Ưu tiên cao nhất
MLP: fc2                       ✓✓           Ưu tiên cao nhất
Classification Head            ✗            Layer cuối
```

### 6.3 Threshold τ — cách chọn

```
Option 1: Target sparsity (recommended để bắt đầu)
    Sắp xếp F_diag(w) tăng dần
    τ = percentile(F_diag, target_binary_ratio)
    Ví dụ: target 80% binary → τ = percentile(F_diag, 80)

Option 2: Thống kê
    τ = mean(F_diag) + c * std(F_diag)
    Tune c trên validation set

Option 3: Per-layer adaptive
    Mỗi layer có τ riêng dựa trên distribution của F_diag layer đó
```

---

## 7. Các quyết định thiết kế cần đưa ra

### Quyết định 1: Scope binarization

```
Lựa chọn A (an toàn nhất): Chỉ FFN layers
    fc1, fc2 của tất cả Swin Blocks
    → ~2/3 params được compress

Lựa chọn B (agressive hơn): FFN + proj
    Thêm attention projection layer

Lựa chọn C (maximum): FFN + proj + qkv
    Cần tách Q/K/V riêng trước khi binarize
```

### Quyết định 2: α và δ trong PTQ setting

```
Option A: Learnable (như APB gốc)
    Ưu: optimal hơn
    Nhược: 1024 images có thể không đủ để converge

Option B: Fixed từ init
    α = mean(|w|), δ = 3*std(w) — không update
    Ưu: stable với ít data
    Nhược: suboptimal

Option C: Chỉ learn α (fix δ)
    Compromise hợp lý
```

### Quyết định 3: Binarize weights only hay cả activations?

```
Recommendation: Weights only (trước)
    Activations sau Softmax và GELU của Swin có distribution đặc biệt
    APB gốc cũng recommend weights-only trước
    → Sau khi weights stable thì mới experiment activations
```

### Quyết định 4: Xử lý QKV

```
Option A: Tách thành 3 Linear riêng [Q_w, K_w, V_w] → per-component threshold
Option B: Giữ nguyên [3C, C] nhưng higher τ (ít binary hơn)
Option C: Không compress qkv (an toàn nhất cho accuracy)
```

---

## 8. Thứ tự implement từng bước

### Bước 1: Verify FIMA-Q baseline

```
□ Clone repo: https://github.com/ShiheWang/FIMA-Q
□ Chạy Swin-S với W4/A4 → verify match Table 1 paper (81.82%)
□ Chạy Swin-S với W3/A3 → verify match (77.26%)
□ Hiểu rõ cấu trúc code: block definition, calibration loop, loss computation
```

### Bước 2: Extract FIM importance per weight

```
□ Trong reconstruction loop của FIMA-Q, sau khi có ∇L_KL và Δz:
    F_diag(z) = ∇L_KL / Δz    (element-wise)
□ Propagate xuống weight:
    F_diag(w_ij) ≈ Σ_k F_diag(z_k) * (∂z_k/∂w_ij)²
                            ↑ có từ squared gradient
□ Lưu F_diag(w) cho mỗi layer vào dict
□ Visualize distribution của F_diag để hiểu data
```

### Bước 3: Implement APB module

```
□ Class APBLayer(nn.Module):
    - forward(): apply APB operator với mask từ F_diag hoặc magnitude
    - compute_alpha_grad(): gradient cho α
    - compute_delta_grad(): gradient cho δ
    - memory_format(): decompose W_bin + W_sparse
□ Test APB layer độc lập với dummy weights
□ Verify gradient flow qua STE
```

### Bước 4: Tích hợp vào FIMA-Q

```
□ Sau FIMA-Q reconstruction → có quantized block + F_diag(w)
□ Apply APB partition lên FFN layers trước
□ Reconstruction lại với DPLR-FIM loss (giờ loss tính trên APB-ized weights)
□ Đo accuracy: Swin-S sau APB FFN-only
```

### Bước 5: Mở rộng và tune

```
□ Mở rộng APB sang proj layers
□ Thử tách QKV và apply APB cho V projection
□ Tune threshold τ (target sparsity 70%, 80%, 90%)
□ Tune α trong DPLR loss
□ So sánh với FIMA-Q alone (baseline)
```

### Bước 6: Evaluation

```
□ ImageNet classification: Swin-S và Swin-B
□ COCO detection (nếu có thời gian): Mask R-CNN với Swin backbone
□ Đo: Top-1 accuracy, memory size, inference time
□ Ablation: FIM-threshold vs magnitude-threshold
```

---

## Appendix: Key Formulas tóm tắt

### FIMA-Q

| Formula | Ý nghĩa |
|---------|---------|
| $\mathcal{L}_{KL} = \frac{1}{2}\Delta z^\top \mathbf{F} \Delta z$ | KL loss = FIM quadratic form |
| $\nabla \mathcal{L}_{KL} = \mathbf{F} \Delta z$ | Gradient = FIM × perturbation |
| $F_{diag,i} = \frac{(\nabla \mathcal{L}_{KL})_i}{\Delta z_i}$ | Diagonal FIM estimation |
| $\mathcal{L}_{DPLR} = \alpha \mathcal{L}_{rank-k} + (1-\alpha)\mathcal{L}_{diag}$ | Final loss |

### APB

| Formula | Ý nghĩa |
|---------|---------|
| $\hat{w} = \frac{\|w\| - \alpha}{\delta}$ | Normalized weight |
| $\chi_B = \mathbf{1}(\hat{w} \leq 1)$ | Binary mask |
| $\frac{\partial \mathcal{L}}{\partial \alpha} = -\frac{1}{\delta n}\sum \frac{\partial \mathcal{L}}{\partial \hat{w}_i} \chi_B$ | Gradient of α |
| $\frac{\partial \mathcal{L}}{\partial \delta} = \frac{1}{\delta^2 n}\sum \frac{\partial \mathcal{L}}{\partial \hat{w}_i}(\alpha - \|w_i\|)\chi_B$ | Gradient of δ |
| $W = W^{bin} + W^{sparse}$ | Memory decomposition |

### Bridge (FIM từ output xuống weight)

| Formula | Ý nghĩa |
|---------|---------|
| $F_{diag}(w_{ij}) \approx \sum_k F_{diag}(z_k) \cdot \left(\frac{\partial z_k}{\partial w_{ij}}\right)^2$ | Propagate FIM xuống weight level |

---

*Tài liệu này tổng hợp từ: FIMA-Q (arXiv:2506.11543v1) + APB (arXiv:2306.08960v2) + phân tích kiến trúc Swin Transformer.*
