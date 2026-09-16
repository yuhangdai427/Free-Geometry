#!/bin/bash
# DA3 7scenes v2 重跑（tap对齐+camera解冻管线）: C2M + pw0，baseline 不重跑
set -u
cd /root/autodl-tmp/Free-Geometry
OUT=artifacts/diagnostics/final_protocol/da3_baseline
WS=workspace/da3_protocol_7scenes_v2
LOG=$OUT/7scenes_v2_loop.log
mkdir -p "$OUT"
echo "=== DA3 7scenes v2 C2M start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/train_da3_protocol.py --dataset 7scenes \
    --scenes chess fire heads office pumpkin redkitchen stairs \
    --output_root "$WS" --steps 100 >> "$LOG" 2>&1 || { echo "TTA C2M FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 7scenes v2 pw0 start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/train_da3_protocol.py --dataset 7scenes \
    --scenes chess fire heads office pumpkin redkitchen stairs \
    --output_root "${WS}_pw0" --steps 100 --pose_weight 0.0 >> "$LOG" 2>&1 || { echo "TTA pw0 FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 7scenes v2 loop DONE $(date '+%F %T') ===" >> "$LOG"
