# NenVIT — APB + FIMA-Q QAT on Swin-S

Nén Swin Transformer cho Tiny ImageNet bằng **APB binarization** ([arXiv:2306.08960](https://arxiv.org/abs/2306.08960)),
chọn weight binary/FP bằng **DPLR-FIM importance** từ FIMA-Q ([arXiv:2506.11543](https://arxiv.org/abs/2506.11543)).
QAT end-to-end với CE loss + optional DPLR distillation loss.

**Novelty:** FIM-based mask thay magnitude, áp lần đầu cho Vision Transformer.

> ⛔ **DEPRECATED (2026-07-13):** `dplr`/`fim` cho **importance/partition** đã bỏ — chỉ dùng **`fisher`** (FIM
> chuẩn) để rank. `dplr` (tự chế) chỉ còn trong **DPLR *loss*** (distillation), KHÔNG dùng ranking. README này
> còn nhiều số/khái niệm cũ (Tiny ImageNet, dplr...) — tin `runs/README.md` + `ABLATIONS.md`.

## Run

```bash
# Quick test
python apb_fimaq/qat.py --debug

# Full
python apb_fimaq/qat.py --use-dplr-loss --dplr-lambda 3000

# Hoặc launcher với preset config
python apb_fimaq/run_full.py
```

## Cấu trúc

```
NenVIT/
├── apb_fimaq/qat.py            Main pipeline (self-contained ~600 LOC)
├── apb_fimaq/run_full.py       Launcher 1-lệnh với preset
├── scripts/tiny_imagenet_loader.py    Tiny ImageNet via HF
└── *.pdf                       APB + FIMA-Q papers
```

## Pipeline

```
1. Load Swin-S pretrained
2. Compute DPLR-FIM(W) — k=5 batches forward+backward
3. Apply APB: mask = (FIM < 75% percentile), wrap nn.Linear → APBLinear
4. QAT 10-20 epochs: loss = CE + λ · Σ_blocks L_DPLR
   - latent_weight + α học; mask FROZEN; freeze α ở epoch/2
```

## APB Linear

```python
binary zone (mask=True):  α · sign(latent_weight)    # 1 bit
FP zone     (mask=False): latent_weight              # 32 bit
```

Backward: custom STE preserves cả `latent_weight` và `α` gradient.

## Target layers

- **`--apb-scope skip`** (default): 96 Linears (qkv/proj/fc1/fc2 × 24), skip head + downsample + patch_embed → 95.1% params
- **`--apb-scope full`**: 100 Linears, thêm head.fc + downsample.reduction → 98.2%

## Key CLI flags

| Flag | Default | |
|---|---|---|
| `--binary-ratio` | 0.75 | % weights binarize |
| `--apb-scope` | skip | skip (96) hoặc full (100) layers |
| `--epochs` | 10 | QAT epochs |
| `--lr` | 1e-4 | AdamW |
| `--batch-size` | 64 | |
| `--fim-batches` | 5 | k calib batches cho importance (+ rank k của DPLR loss) |
| `--importance-full` | False | tính importance trên **toàn dataset** (1 pass, exact; nên dùng với `fisher`) |
| `--logits-reversal` | False | LR (arXiv 2603.18596): dùng CE(-logit) khi tính importance, fix gradient-vanishing; trực giao với `--importance` (`fisher`+LR = Ω^LR của paper) |
| `--importance` | fisher | ⛔ chỉ `fisher` (dplr đã bỏ; `--fim-mode` alias) |
| `--use-dplr-loss` | False | DPLR distillation loss |
| `--dplr-lambda` | 1.0 | Dùng **3000** (raw DPLR ~0.002 vs CE ~6) |
| `--debug` | False | 2-epoch smoke test |

## Design quan trọng

- **Mask FROZEN** (không recompute) → stability. Verified: 0% mask flip, 0.25% sign flip.
- **α learnable** với custom STE (paper Eq 8: `∂L/∂α = Σ grad·sign(w)·χ_binary`).
- **Weight decay** chỉ áp lên weights, **KHÔNG áp lên α** (paper APB rule).
- **DPLR-FIM** dùng 2 chỗ:
  - Importance ranking (per-weight, 1 lần) → set mask
  - Loss distillation (per-block per-batch) → guide gradient

## Dataset

Tiny ImageNet (200 classes 64×64) via HuggingFace `zh-plus/tiny-imagenet`.
18 classes outside ImageNet-1k auto-filter → 182 classes mapped to 1000 indices → dùng head pretrained trực tiếp.

## Setup

```bash
conda create -n fimaq_apb python=3.10 -y && conda activate fimaq_apb
# Blackwell (RTX 50xx):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Other GPUs (Kaggle, Ampere, ...):
pip install torch torchvision
pip install datasets timm
```

## Kaggle

```python
!git clone https://github.com/xauskeleton/NenVIT.git && cd NenVIT
!pip install -q datasets timm
!python apb_fimaq/qat.py --use-dplr-loss --dplr-lambda 3000 \
    --batch-size 32 --out-dir /kaggle/working/qat
```

## Expected results

| Stage | Top-1 Tiny val |
|---|---|
| FP baseline | 55.76% |
| Post-APB (no train) | ~5% |
| 2-ep debug | ~32% |
| 20-ep target | **45-55%** |

Paper FIMA-Q báo 81.82% Swin-S W4A4 trên full ImageNet-1k (task khác).

---

Repo: https://github.com/xauskeleton/NenVIT
