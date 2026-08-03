#!/usr/bin/env bash
# =============================================================================
# Ablation A1 + (A2×A3×act) cho Swin-S, CHUẨN HÓA về config train-thật:
#   50ep · batch 64 · patience 15 · seed 3407 · br0.99 · DPLR λ0.1
# Chạy trên MODEL PRUNED (ckpt_swin/pruned/best_pruned_model.pt) — cùng base với
# headline multi-arch, reuse được 2 ô đã có từ e2e:
#   magnitude+DPLR+act1 = ckpt_swin/prune_quant_br0.99_act1 (52.47%)
#   magnitude+DPLR+act2 = ckpt_swin/prune_quant_br0.99_act2 (81.23%)
#
# Lưới đầy đủ = partition{magnitude,fisher} × DPLR{on,off} × act{1,2} = 2×2×2 = 8 ô.
# Script này chạy 6 ô CÒN THIẾU + 1 ô A1 (magnitude-prune). Tự skip nếu đã có best.pth.
#
# USAGE (trên GPU, đã conda activate fimaq):
#   bash scripts/run_ablation_swin.sh
#   ACT_LIST="2" bash scripts/run_ablation_swin.sh   # chỉ nhánh act2 (nếu ít thời gian)
#   PART_LIST="fisher" bash scripts/run_ablation_swin.sh
# =============================================================================
set -euo pipefail

PY="${PY:-python}"
export PYTHONUTF8=1
MODEL="swin_small_patch4_window7_224"
DATASET="cifar100"
SEED="${SEED:-3407}"
BR="${BR:-0.99}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
PATIENCE="${PATIENCE:-15}"
DPLR_LAMBDA="${DPLR_LAMBDA:-0.1}"   # loss chuan hoa O(1) tu 2026-08-02 (ban cu: 3000)

PRUNED="ckpt_swin/pruned/best_pruned_model.pt"          # fisher-0.5 (A1 arm 1, đã có)
BASELINE="ckpt_swin/ft/best.pth"                         # FP baseline cho prune
ABL="ckpt_swin/ablation"                                 # gom mọi ô ablation
mkdir -p "$ABL"

# ô muốn quét (override qua env)
PART_LIST="${PART_LIST:-magnitude fisher}"
DPLR_LIST="${DPLR_LIST:-on off}"
ACT_LIST="${ACT_LIST:-1 2}"

echo "=============================================================="
echo " ABLATION Swin | br=$BR ep=$EPOCHS batch=$BATCH patience=$PATIENCE seed=$SEED"
echo " grid: part[$PART_LIST] × dplr[$DPLR_LIST] × act[$ACT_LIST]  (base=pruned)"
echo "=============================================================="

# ---- A1: magnitude-prune (fisher arm = ckpt_swin/pruned đã có) ----
# Chạy cả l1 (chuẩn Li et al. filter-pruning) lẫn l2sq (baseline magnitude mạnh hơn).
MAG_NORM_LIST="${MAG_NORM_LIST:-l1 l2sq}"
for norm in $MAG_NORM_LIST; do
  A1OUT="$ABL/A1_prune_magnitude_${norm}_g05"
  if [ -f "$A1OUT/best_pruned_model.pt" ]; then
    echo ">> A1 magnitude($norm)-prune SKIP (đã có $A1OUT)"
    continue
  fi
  echo ">> A1: prune magnitude ($norm, global 0.5) -> $A1OUT"
  "$PY" apb_fimaq/prune_cifar.py --model "$MODEL" --dataset "$DATASET" \
    --baseline-ckpt "$BASELINE" \
    --importance magnitude --mag-norm "$norm" --rank-mode global --global-metric per_param \
    --head-ratio 0.5 --mlp-ratio 0.5 --mlp-keep-frac 0.05 \
    --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$SEED" --patience "$PATIENCE" \
    --out-dir "$A1OUT"
done

# ---- A2×A3×act: lưới 2×2×2 trên model pruned ----
for part in $PART_LIST; do
  for dplr in $DPLR_LIST; do
    for act in $ACT_LIST; do
      TAG="${part}_${dplr}dplr_act${act}"
      OUT="$ABL/$TAG"

      # reuse 2 ô e2e: magnitude+on+act{1,2} đã nằm ở prune_quant_br0.99_act*
      if [ "$part" = "magnitude" ] && [ "$dplr" = "on" ]; then
        SRC="ckpt_swin/prune_quant_br${BR}_act${act}"
        echo ">> SKIP $TAG — reuse ô e2e đã có: $SRC"
        continue
      fi
      if [ -f "$OUT/best.pth" ]; then
        echo ">> SKIP $TAG — đã có $OUT/best.pth"
        continue
      fi

      DPLR_FLAGS=""
      [ "$dplr" = "on" ] && DPLR_FLAGS="--use-dplr-loss --dplr-lambda $DPLR_LAMBDA"

      echo ">> RUN $TAG -> $OUT"
      "$PY" apb_fimaq/qat.py --model "$MODEL" --dataset "$DATASET" \
        --init-model "$PRUNED" \
        --partition "$part" --binary-ratio "$BR" --act-bits "$act" \
        $DPLR_FLAGS \
        --epochs "$EPOCHS" --batch-size "$BATCH" --seed "$SEED" --patience "$PATIENCE" \
        --out-dir "$OUT"
    done
  done
done

echo "=============================================================="
echo " DONE ablation. Kết quả: grep 'Best val top1' $ABL/*/train.log"
echo " Reuse: ckpt_swin/prune_quant_br${BR}_act1 (mag+DPLR+act1), _act2 (mag+DPLR+act2)"
echo "=============================================================="
