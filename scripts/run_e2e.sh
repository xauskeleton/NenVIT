#!/usr/bin/env bash
# =============================================================================
# End-to-end pipeline: FP baseline -> structural prune (fisher) -> APB-QAT
# (magnitude partition + DPLR distillation), on CIFAR-100.
#
# Chạy full cho MỘT model (Swin hoặc ViT/DeiT). Dùng đúng default đã chốt:
#   prune  --importance fisher   |   qat --partition magnitude   |   DPLR λ=0.1
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
DPLR_LAMBDA="${DPLR_LAMBDA:-0.1}"   # loss da chuan hoa O(1) tu 2026-08-02; ban cu can 3000
QTAG="${QTAG:-}"                    # hau to out-dir, de khong de len ket qua cu
# training
FT_EPOCHS="${FT_EPOCHS:-50}"
PRUNE_EPOCHS="${PRUNE_EPOCHS:-50}"
QAT_EPOCHS="${QAT_EPOCHS:-50}"   # epoch cao để cosine LR (T_max=epochs) anneal hết về ~0 ở cuối
BATCH="${BATCH:-32}"
PATIENCE="${PATIENCE:-15}"  # early-stop patience cho CẢ 3 stage — CAO, ghép với epoch cao (50).
                            # Lý do: cosine LR (T_max=epochs) chỉ anneal ~0 ở epoch cuối. patience=5 cũ
                            # cắt quá sớm khi LR còn cao → prune cost phồng (Swin ep12/50 LR86% cost2.06%;
                            # ViT ep17/50 cost4.10%; DeiT ep21/50 cost2.22%). patience=15 + 50ep: nửa sau
                            # cosine hạ LR → val bò lên tạo best mới → reset counter → tự chạy gần hết,
                            # early-stop chỉ cắt run thật sự chết. Muốn tuyệt đối chắc: PATIENCE=0.

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

PRUNED="$OUT/pruned/best_pruned_model.pt"
# Stage 1 chỉ phục vụ Stage 2. Có pruned model rồi thì cả hai đều thừa — đừng
# finetune 50 epoch chỉ để rồi vứt đi (đúng bẫy đã dính khi server bị xoá sạch).
SKIP_FT_PRUNE=0
if [ "${FORCE:-0}" != "1" ] && [ -f "$PRUNED" ]; then SKIP_FT_PRUNE=1; fi

# ---- Stage 1: FP baseline (auto-skip nếu đã có; FORCE=1 để chạy lại) ----
BASELINE="${BASELINE:-}"
if [ "$SKIP_FT_PRUNE" = "1" ]; then
  echo ">> Stage 1 SKIP — không cần baseline, đã có pruned model"
elif [ -z "$BASELINE" ]; then
  if [ "$MODEL" = "swin_small_patch4_window7_224" ] && [ -f ckpt/best_swin.pth ]; then
    BASELINE="ckpt/best_swin.pth"
    echo ">> Stage 1 SKIP — tái dùng baseline Swin có sẵn: $BASELINE"
  elif [ "${FORCE:-0}" != "1" ] && [ -f "$OUT/ft/best.pth" ]; then
    BASELINE="$OUT/ft/best.pth"
    echo ">> Stage 1 SKIP — baseline đã có: $BASELINE  (FORCE=1 để chạy lại)"
  else
    echo ">> Stage 1: finetune FP baseline -> $OUT/ft/best.pth"
    "$PY" apb_fimaq/finetune.py --model "$MODEL" --dataset "$DATASET" \
      --epochs "$FT_EPOCHS" --batch-size "$BATCH" --seed "$SEED" --patience "$PATIENCE" \
      --out-dir "$OUT/ft" $DBG
    BASELINE="$OUT/ft/best.pth"
  fi
fi

# ---- Stage 2: structural prune (fisher) — auto-skip nếu pruned model đã có ----
if [ "$SKIP_FT_PRUNE" = "1" ]; then
  echo ">> Stage 2 SKIP — pruned model đã có: $PRUNED  (FORCE=1 để chạy lại)"
else
  echo ">> Stage 2: prune (fisher, $RANK_MODE) -> $PRUNED"
  "$PY" apb_fimaq/prune_cifar.py --model "$MODEL" --dataset "$DATASET" \
    --baseline-ckpt "$BASELINE" \
    --importance fisher --rank-mode "$RANK_MODE" --global-metric per_param \
    --head-ratio "$HEAD_RATIO" --mlp-ratio "$MLP_RATIO" --mlp-keep-frac "$MLP_KEEP_FRAC" \
    --epochs "$PRUNE_EPOCHS" --batch-size "$BATCH" --seed "$SEED" --patience "$PATIENCE" \
    --out-dir "$OUT/pruned" $DBG
fi

# ---- Stage 3: APB-QAT trên model đã prune (magnitude + DPLR) ----
# out-dir gồm br/act → chạy e2e nhiều lần với BR/ACT khác nhau KHÔNG đè nhau
# (baseline+prune auto-skip vì đã có; chỉ quant chạy lại → 1 prune, nhiều quant).
QOUT="$OUT/prune_quant_br${BR}_act${ACT_BITS}${QTAG}"
if [ -f "$QOUT/best.pth" ] && [ "${FORCE:-0}" != "1" ]; then
  echo ">> Stage 3 SKIP — da co: $QOUT/best.pth  (FORCE=1 de chay lai)"
  grep -ah "DONE. Best val top1" "$QOUT/train.log" 2>/dev/null || true
  exit 0
fi
echo ">> Stage 3: APB-QAT (magnitude+DPLR) -> $QOUT/best.pth"
"$PY" apb_fimaq/qat.py --model "$MODEL" --dataset "$DATASET" \
  --init-model "$PRUNED" \
  --partition magnitude --binary-ratio "$BR" --act-bits "$ACT_BITS" \
  --use-dplr-loss --dplr-lambda "$DPLR_LAMBDA" \
  --epochs "$QAT_EPOCHS" --batch-size "$BATCH" --seed "$SEED" --patience "$PATIENCE" \
  --out-dir "$QOUT" $DBG

echo "=============================================================="
echo " DONE. Kết quả cuối: $QOUT/best.pth (+ best_packed.pt)"
echo " Baseline: $BASELINE | Pruned: $PRUNED"
echo "=============================================================="
