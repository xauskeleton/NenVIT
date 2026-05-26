# NenVIT — APB + FIMA-Q on Swin Transformer

Combining **Automatic Prune Binarization (APB)** with **FIMA-Q** (Fisher Information Matrix Approximation for PTQ) to compress Swin Transformer.

**Core idea:** Use FIMA-Q's FIM importance scores (instead of magnitude) to decide which weights to binarize (±α) vs keep full-precision, then re-optimize with FIMA-Q's DPLR-FIM block reconstruction loss.

## Papers

- **APB** — Nardini et al. *Neural Network Compression using Binarization and Few Full-Precision Weights*, arXiv:2306.08960v2
- **FIMA-Q** — Wu et al. *FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation*, CVPR 2025, arXiv:2506.11543v1

PDFs in repo for convenience.

## Repo structure

```
.
├── apb_fimaq/                            ← OUR contribution (the APB extension)
│   ├── apb_weight_quantizer.py           Core APB(w)=α·sign(w) with STE backward
│   ├── fim_weight_extractor.py           Bridge F_diag(z) → F_diag(W)
│   ├── apb_wrap.py                       Filter 96 target Linears + replace w_quantizer
│   ├── run_phase_a.py                    Phase A: standard FIMA-Q W4A4 baseline
│   ├── run_apb_pipeline.py               Full pipeline A→B→C→D
│   ├── debug_pipeline.py                 Fast smoke test
│   └── test_apb_weight_quantizer.py      Unit tests (9/9 pass)
│
├── scripts/                              ← Analysis + dataloader
│   ├── tiny_imagenet_loader.py           HF Tiny ImageNet → FIMA-Q's LoaderGenerator
│   ├── 02_smoketest_swin.py              FP baseline eval
│   ├── dump_swin_*.py                    Inspection scripts
│   ├── analyze_*.py                      Per-layer quantizability analysis
│   └── *.txt, *.csv                      Generated reports
│
├── FIMA-Q/FIMA-Q/                        ← Upstream FIMA-Q (UNTOUCHED)
│
├── APB_FIMAQ_Swin_Implementation_Guide.md
├── 2306.08960v2 (1).pdf                  APB paper
└── 2506.11543v1.pdf                      FIMA-Q paper
```

## Pipeline overview

| Phase | What it does | Code |
|---|---|---|
| **A** | Standard FIMA-Q: wrap → calibrate → DPLR-FIM block reconstruction → W4A4 baseline | uses upstream FIMA-Q |
| **B** | Extract `F_diag(W_ij) ≈ E[\|∇z_i\|] · E[x_j²]` per APB target Linear (96 layers) | `fim_weight_extractor.py` |
| **C** | Replace each target's `w_quantizer` with `APBWeightQuantizer`; mask = (FIM < τ) | `apb_wrap.py` |
| **D** | Re-run block reconstruction; only activation scales fine-tune (APB partition frozen) | reuses FIMA-Q (TODO: subclass to skip AdaRound re-wrap on APB layers) |

## APB target selection

**96 Linears** out of 100 (qkv, attn.proj, mlp.fc1, mlp.fc2 across all 24 Swin blocks).

**Skipped (4):** 3 `downsample.reduction` + `head.fc`. Also skipped: `patch_embed.proj` (Conv2d), all 53 LayerNorm.

→ Coverage: 47.19M / 49.61M params = **95.1% of the model**.

## Setup

```bash
# Create env (Python 3.10)
conda create -n fimaq_apb python=3.10 -y
conda activate fimaq_apb

# Install PyTorch (CUDA 12.8 for RTX 50-series Blackwell support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other deps
pip install datasets timm scipy matplotlib
```

## Dataset

Uses **Tiny ImageNet** (200 classes, 64×64) via Hugging Face for debug/dev. 18 classes outside ImageNet-1k are auto-filtered → 182 classes mapped to ImageNet-1k indices so pretrained Swin works directly. See `scripts/tiny_imagenet_loader.py`.

## Run

```bash
# Quick debug (~1 min end-to-end with tiny sizes)
python apb_fimaq/debug_pipeline.py

# Phase A only (FIMA-Q baseline W4A4)
python apb_fimaq/run_phase_a.py --calib-size 64 --optim-size 256

# Full pipeline with default 75% binary ratio
python apb_fimaq/run_apb_pipeline.py

# Per-role binary ratio (conservative for attention)
python apb_fimaq/run_apb_pipeline.py \
    --ratio-qkv 0.60 \
    --ratio-proj 0.75 \
    --ratio-fc1 0.85 \
    --ratio-fc2 0.75

# Reuse calibrated checkpoint
python apb_fimaq/run_apb_pipeline.py \
    --load-optimize-ckpt checkpoints/phase_a/phase_a_optimized.pth
```

## Status

- ✅ Phase A wrap + calibration verified on Tiny ImageNet (smoke test passes)
- ✅ Phase B FIM extraction (after fixing mode-switching and STE-mode bugs)
- ✅ Phase C APB swap (96 layers, runs without crash)
- ⚠️ Phase D needs subclass of `BlockReconstructor` to skip AdaRound re-wrap on APB layers

## Design principle

**Do not modify FIMA-Q.** Subclass or copy-then-modify in `apb_fimaq/` instead. This keeps the upstream pristine for reproducibility and clean diffs.

## Hardware tested

- NVIDIA RTX 5060 Ti (16GB, Blackwell sm_120)
- CUDA 12.8 / PyTorch 2.11.0

---

*Project for NCKH (research). Author: xauskeleton*
