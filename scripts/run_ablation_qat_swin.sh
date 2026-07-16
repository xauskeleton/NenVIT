#!/usr/bin/env bash
# =============================================================================
# Ablation QAT (A2×A3×act) cho Swin-S — CHỈ CHẠY QAT, KHÔNG prune.
#   Lưới = partition{magnitude,fisher} × DPLR{on,off} × act{1,2} = 2×2×2 = 8 ô.
#   Config train-thật: 50ep · batch 64 · patience 15 · seed 3407 · br0.99 · DPLR λ3000.
#   Base = model pruned SẴN CÓ (ckpt_swin/pruned = fisher-0.5) — chỉ ĐỌC, không ghi.
#
# AN TOÀN FILE: chỉ ghi vào ckpt_swin/ablation/qat_*  — KHÔNG đụng
#   ckpt_swin/{ft,pruned,prune_quant_*}. Idempotent: có best.pth thì skip.
#
# Reuse 2 ô e2e đã có (magnitude+DPLR+act{1,2} = 52.47 / 81.23) -> chỉ chạy 6 ô còn thiếu.
#
# USAGE (trên GPU, đã conda activate fimaq):
#   bash scripts/run_ablation_qat_swin.sh
#   ACT_LIST="2" bash scripts/run_ablation_qat_swin.sh     # chỉ nhánh act2 (nhanh)
#   PART_LIST="fisher" bash scripts/run_ablation_qat_swin.sh
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
DPLR_LAMBDA="${DPLR_LAMBDA:-3000}"

PRUNED="ckpt_swin/pruned/best_pruned_model.pt"   # fisher-0.5, base chung — CHỈ ĐỌC
ABL="ckpt_swin/ablation"
mkdir -p "$ABL"

PART_LIST="${PART_LIST:-magnitude fisher}"
DPLR_LIST="${DPLR_LIST:-on off}"
ACT_LIST="${ACT_LIST:-1 2}"

if [ ! -f "$PRUNED" ]; then
  echo "LỖI: không thấy model pruned '$PRUNED' — cần stage prune xong trước."; exit 1
fi

echo "=============================================================="
echo " ABLATION QAT Swin | base=$PRUNED"
echo " grid: part[$PART_LIST] × dplr[$DPLR_LIST] × act[$ACT_LIST]"
echo " br=$BR ep=$EPOCHS batch=$BATCH patience=$PATIENCE seed=$SEED"
echo "=============================================================="

for part in $PART_LIST; do
  for dplr in $DPLR_LIST; do
    for act in $ACT_LIST; do
      TAG="qat_${part}_${dplr}dplr_act${act}"
      OUT="$ABL/$TAG"

      # ô magnitude+DPLR+act{1,2} đã có từ e2e -> reuse, KHÔNG chạy lại
      if [ "$part" = "magnitude" ] && [ "$dplr" = "on" ]; then
        echo ">> SKIP $TAG — reuse ckpt_swin/prune_quant_br${BR}_act${act} (đã có)"
        continue
      fi
      if [ -f "$OUT/best.pth" ]; then
        echo ">> SKIP $TAG — đã xong ($OUT/best.pth)"
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
echo " DONE. Kết quả 6 ô mới:  grep 'Best val top1' $ABL/qat_*/train.log"
echo "2 ô reuse: ckpt_swin/prune_quant_br${BR}_act1 (52.47) / _act2 (81.23)"
echo "=============================================================="
