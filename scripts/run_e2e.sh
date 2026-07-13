#!/usr/bin/env bash
# =============================================================================
# End-to-end pipeline: FP baseline -> structural prune (fisher) -> APB-QAT
# (magnitude partition + DPLR distillation), on CIFAR-100.
#
# Chạy full cho MỘT model (Swin hoặc ViT/DeiT). Dùng đúng default đã chốt:
#   prune  --importance fisher   |   qat --partition magnitude   |   DPLR λ=3000
#
# USAGE (mọi tham số qua biến môi trường, có default):
#   bash scripts/run_e2e.sh                       # Swin-S (tái dùng ckpt/best_swin.pth)
#   MODEL=deit_small_patch16_224 bash scripts/run_e2e.sh
#   MODEL=vit_small_patch16_224 BR=0.99 ACT_BITS=2 bash scripts/run_e2e.sh
#   QUICK=1 bash scripts/run_e2e.sh               # smoke test (--debug mọi stage)
#   BASELINE=ckpt/best_swin.pth bash scripts/run_e2e.sh   # bỏ qua finetune, dùng baseline có sẵn
#
# Kết quả cuối in ở stage 3 (best top-1) + lưu tại $OUT/prune_quant/best.pth (+ best_packed.pt).
# =============================================================================
set -euo pipefail

# ---- config (override qua env) ----
# PY mặc định = 'python' của conda env ĐANG ACTIVATE (portable Windows/Linux).
#   Windows:  conda activate fimaq_apb   (hoặc override PY=C:/.../python.exe)
#   Linux:    conda activate fimaq
PY="${PY:-python}"
MODEL="${MODEL:-swin_small_patch4_window7_224}"
DATASET="${DATASET:-cifar100}"
SEED="${SEED:-3407}"
export PYTHONUTF8=1                     # console Windows cp1252 không encode được 'α'

# prune knobs (fisher là default trong prune_cifar.py)
HEAD_RATIO="${HEAD_RATIO:-0.5}"
MLP_RATIO="${MLP_RATIO:-0.5}"
MLP_KEEP_FRAC="${MLP_KEEP_FRAC:-0.05}"
RANK_MODE="${RANK_MODE:-global}"
# quant knobs (magnitude + scope=full là default trong qat.py)
BR="${BR:-0.95}"
ACT_BITS="${ACT_BITS:-2}"
DPLR_LAMBDA="${DPLR_LAMBDA:-3000}"
# training
FT_EPOCHS="${FT_EPOCHS:-20}"
PRUNE_EPOCHS="${PRUNE_EPOCHS:-20}"
QAT_EPOCHS="${QAT_EPOCHS:-30}"   # chuẩn hóa 30 (khớp C5-C8; số cũ 29 chỉ ở vài run combined)
BATCH="${BATCH:-32}"

# out-dir suy từ tên họ model: ckpt_swin / ckpt_deit / ckpt_vit ...
OUT="${OUT:-ckpt_${MODEL%%_*}}"
mkdir -p "$OUT"

# QUICK=1 -> thêm --debug vào mọi stage (smoke, vài batch)
DBG=""
[ "${QUICK:-0}" = "1" ] && DBG="--debug"

echo "=============================================================="
echo " E2E | model=$MODEL | dataset=$DATASET | out=$OUT"
echo " prune: fisher/$RANK_MODE head=$HEAD_RATIO mlp=$MLP_RATIO"
echo " quant: scope=full br=$BR act=$ACT_BITS dplr_lambda=$DPLR_LAMBDA"
echo " quick=${QUICK:-0}"
echo "=============================================================="

# ---- Stage 1: FP baseline (auto-skip nếu đã có; FORCE=1 để chạy lại) ----
BASELINE="${BASELINE:-}"
if [ -z "$BASELINE" ]; then
  if [ "$MODEL" = "swin_small_patch4_window7_224" ] && [ -f ckpt/best_swin.pth ]; then
    BASELINE="ckpt/best_swin.pth"
    echo ">> Stage 1 SKIP — tái dùng baseline Swin có sẵn: $BASELINE"
  elif [ "${FORCE:-0}" != "1" ] && [ -f "$OUT/ft/best.pth" ]; then
    BASELINE="$OUT/ft/best.pth"
    echo ">> Stage 1 SKIP — baseline đã có: $BASELINE  (FORCE=1 để chạy lại)"
  else
    echo ">> Stage 1: finetune FP baseline -> $OUT/ft/best.pth"
    "$PY" apb_fimaq/finetune.py --model "$MODEL" --dataset "$DATASET" \
      --epochs "$FT_EPOCHS" --batch-size "$BATCH" --seed "$SEED" \
      --out-dir "$OUT/ft" $DBG
    BASELINE="$OUT/ft/best.pth"
  fi
fi

# ---- Stage 2: structural prune (fisher) — auto-skip nếu pruned model đã có ----
PRUNED="$OUT/pruned/best_pruned_model.pt"
if [ "${FORCE:-0}" != "1" ] && [ -f "$PRUNED" ]; then
  echo ">> Stage 2 SKIP — pruned model đã có: $PRUNED  (FORCE=1 để chạy lại)"
else
  echo ">> Stage 2: prune (fisher, $RANK_MODE) -> $PRUNED"
  "$PY" apb_fimaq/prune_cifar.py --model "$MODEL" --dataset "$DATASET" \
    --baseline-ckpt "$BASELINE" \
    --importance fisher --rank-mode "$RANK_MODE" --global-metric per_param \
    --head-ratio "$HEAD_RATIO" --mlp-ratio "$MLP_RATIO" --mlp-keep-frac "$MLP_KEEP_FRAC" \
    --epochs "$PRUNE_EPOCHS" --batch-size "$BATCH" --seed "$SEED" \
    --out-dir "$OUT/pruned" $DBG
fi

# ---- Stage 3: APB-QAT trên model đã prune (magnitude + DPLR) ----
echo ">> Stage 3: APB-QAT (magnitude+DPLR) -> $OUT/prune_quant/best.pth"
"$PY" apb_fimaq/qat.py --model "$MODEL" --dataset "$DATASET" \
  --init-model "$PRUNED" \
  --partition magnitude --binary-ratio "$BR" --act-bits "$ACT_BITS" \
  --use-dplr-loss --dplr-lambda "$DPLR_LAMBDA" \
  --epochs "$QAT_EPOCHS" --batch-size "$BATCH" --seed "$SEED" \
  --out-dir "$OUT/prune_quant" $DBG

echo "=============================================================="
echo " DONE. Kết quả cuối: $OUT/prune_quant/best.pth (+ best_packed.pt)"
echo " Baseline: $BASELINE | Pruned: $PRUNED"
echo "=============================================================="
