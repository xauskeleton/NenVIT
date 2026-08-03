#!/usr/bin/env bash
# =============================================================================
# QAT với DPLR loss đã sửa — chạy MỘT ô để so với số cũ.
#
# Default khớp đúng ô legacy đã có trong runs/README.md
# (checkpoints/partition_dplr_grid/magnitude_ondplr_act2 = 81.23%):
#   init = model đã prune | br0.99 | act2 | magnitude | 50ep | bs64 | seed3407
# Khác biệt duy nhất là bản thân DPLR loss.
#
# USAGE (mọi tham số qua biến môi trường, có default):
#   bash scripts/run_dplr_e.sh                    # loss mới, λ=0.1
#   ARM=legacy bash scripts/run_dplr_e.sh         # loss cũ, λ=3000 (đối chứng)
#   QUICK=1 bash scripts/run_dplr_e.sh            # smoke test (--debug)
#   EPOCHS=30 BATCH=32 bash scripts/run_dplr_e.sh
#   INIT=ckpt_deit/pruned/best_pruned_model.pt OUT=ckpt_deit/dplr_e bash scripts/run_dplr_e.sh
#
# Chạy trong tmux thì tự mở trước:  tmux new -s dplr
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
export PYTHONUTF8=1

ARM="${ARM:-e}"
MODEL="${MODEL:-swin_small_patch4_window7_224}"
DATASET="${DATASET:-cifar100}"
INIT="${INIT:-ckpt_swin/pruned/best_pruned_model.pt}"
QUICK="${QUICK:-0}"
# QUICK ghi ra thư mục riêng, nếu không smoke sẽ để lại best.pth và run thật bị skip
_OUT_DEFAULT="ckpt_swin/dplr_${ARM}_act2"
[ "$QUICK" = "1" ] && _OUT_DEFAULT="${_OUT_DEFAULT}_smoke"
OUT="${OUT:-$_OUT_DEFAULT}"
BR="${BR:-0.99}"
ACT_BITS="${ACT_BITS:-2}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
SEED="${SEED:-3407}"
PATIENCE="${PATIENCE:-15}"
FORCE="${FORCE:-0}"

case "$ARM" in
  e)      ARM_FLAGS=(--dplr-lambda "${LAMBDA:-0.1}") ;;
  legacy) ARM_FLAGS=(--dplr-legacy-loss --dplr-lambda "${LAMBDA:-3000}") ;;
  *)      echo "!! ARM phải là 'e' hoặc 'legacy', nhận được '$ARM'" >&2; exit 2 ;;
esac

DBG=""
[ "$QUICK" = "1" ] && DBG="--debug"

[ -f "$INIT" ] || { echo "!! không thấy init model: $INIT" >&2; exit 1; }
if [ -f "$OUT/best.pth" ] && [ "$FORCE" != "1" ]; then
  echo ">>> $OUT/best.pth đã có — bỏ qua (FORCE=1 để chạy lại)"
  exit 0
fi
mkdir -p "$OUT"

echo "=============================================================="
echo " DPLR QAT | arm=$ARM | $MODEL | $DATASET"
echo " init=$INIT"
echo " br=$BR act=$ACT_BITS ${EPOCHS}ep bs=$BATCH seed=$SEED -> $OUT"
echo " mốc so (loss cũ, cùng ô): 81.23%"
echo "=============================================================="
t0=$SECONDS

"$PY" apb_fimaq/qat.py --model "$MODEL" --dataset "$DATASET" \
  --init-model "$INIT" \
  --binary-ratio "$BR" --partition magnitude --act-bits "$ACT_BITS" \
  --use-dplr-loss "${ARM_FLAGS[@]}" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$SEED" --patience "$PATIENCE" \
  --out-dir "$OUT" $DBG

echo "=============================================================="
echo " xong trong $(( (SECONDS - t0) / 60 )) phút — log: $OUT/train.log"
echo "=============================================================="
