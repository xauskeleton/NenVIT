#!/usr/bin/env bash
# Xem tiến độ + kết quả pipeline 3 model (chạy lúc nào cũng được, kể cả đang train).
# USAGE:  cd ~/NenVIT && bash scripts/status.sh
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || true   # về thư mục repo

echo "==================== FP BASELINE (ft) ===================="
for m in vit deit swin; do
  f=ckpt_$m/ft/train.log
  [ -f "$f" ] && printf "  %-5s : %s\n" "$m" "$(grep 'DONE. Best val top1' "$f" | tail -1 | sed 's/DONE. //')"
done

echo ""
echo "==================== PRUNE ===================="
for m in vit deit swin; do
  f=ckpt_$m/pruned/train.log
  [ -f "$f" ] && printf "  %-5s : %s | %s\n" "$m" \
    "$(grep 'Best pruned val top1' "$f" | tail -1 | sed 's/DONE. //')" \
    "$(grep 'prune cost' "$f" | tail -1)"
done

echo ""
echo "==================== QUANT (config đã xong) ===================="
for f in ckpt_*/prune_quant_*/train.log; do
  [ -f "$f" ] || continue
  done_line=$(grep 'DONE. Best val top1' "$f" | tail -1)
  if [ -n "$done_line" ]; then
    cfg=$(echo "$f" | sed -E 's#ckpt_([a-z]+)/prune_quant_(br[0-9.]+_act[0-9])/.*#\1 \2#')
    effbit=$(grep 'Effective bit (WHOLE' "$f" | tail -1 | grep -oE '[0-9.]+ bits/weight' | head -1)
    printf "  %-22s : %s | %s\n" "$cfg" "$(echo "$done_line" | grep -oE '[0-9.]+%')" "$effbit"
  fi
done

echo ""
echo "==================== ĐANG CHẠY (epoch hiện tại) ===================="
latest=$(ls -t ckpt_*/*/train.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then
  echo "  file: $latest"
  grep -E "^Ep [0-9]+/|Applying APB|>> Stage|DONE" "$latest" | tail -4 | sed 's/^/  /'
else
  echo "  (chưa có log nào)"
fi
