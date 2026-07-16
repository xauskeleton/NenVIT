#!/usr/bin/env bash
# =============================================================================
# Chạy FULL pipeline cho CẢ 3 model (ViT / DeiT / Swin) tuần tự.
# Mỗi model: FP baseline -> prune (fisher) -> APB-QAT (magnitude+DPLR).
# Dùng config đã fix: 50ep + patience 15 (cosine anneal đủ, không cắt sớm).
#
# THỨ TỰ: vit -> deit -> swin (nhanh trước, Swin chậm nhất chạy cuối →
#          có kết quả 2 model sớm; đổi mảng MODELS nếu muốn thứ tự khác).
#
# CONFIG quant: đủ 4 (br0.95/0.99 × act2/act1). act1 chết 48-50% nhưng vẫn chạy để bảng paper đủ.
#   Muốn bỏ act1 cho nhanh: đổi CONFIGS="0.95:2 0.99:2"
#
# USAGE (trong tmux, từ thư mục repo trên GPU):
#   cd ~/NenVIT
#   bash scripts/run_all3.sh 2>&1 | tee ~/run_all3.log
#
# Lần quant đầu mỗi model FORCE=1 (redo ft+prune); lần sau auto-skip ft+prune.
# =============================================================================
set -euo pipefail

export BATCH="${BATCH:-64}"
export PYTHONUTF8=1
# export QAT_EPOCHS=30   # <-- bỏ comment nếu muốn hạ QAT xuống 30ep cho nhanh (~40% giờ quant)

MODELS=(
  vit_small_patch16_224
  deit_small_patch16_224
  swin_small_patch4_window7_224
)
CONFIGS="0.95:2 0.99:2 0.95:1 0.99:1"   # "BR:ACT" — đủ 4 config (act1 vẫn chạy để có bảng đầy đủ)

for M in "${MODELS[@]}"; do
  echo ""
  echo "############################################################"
  echo "#  MODEL: $M"
  echo "############################################################"
  first=1
  for c in $CONFIGS; do
    BR="${c%:*}"; ACT="${c#*:}"
    echo ">>> $M | br=$BR act=$ACT (first=$first)"
    if [ "$first" = "1" ]; then
      # lần đầu: redo finetune + prune (config mới) rồi quant
      FORCE=1 MODEL="$M" BR="$BR" ACT_BITS="$ACT" bash scripts/run_e2e.sh
      first=0
    else
      # các lần sau: tái dùng baseline+prune vừa tạo, chỉ chạy quant
      MODEL="$M" BR="$BR" ACT_BITS="$ACT" bash scripts/run_e2e.sh
    fi
  done
done

echo ""
echo "############################################################"
echo "#  XONG CẢ 3 MODEL. Tổng hợp kết quả:"
echo "############################################################"
grep -H "DONE. Best val top1" ckpt_*/prune_quant_*/train.log 2>/dev/null || true
