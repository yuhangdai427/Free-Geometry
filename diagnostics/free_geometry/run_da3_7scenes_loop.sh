#!/bin/bash
# DA3 7scenes 闭环: baseline(无适配) -> TTA C2M(maskrel) -> 结果对比
set -u
cd /root/autodl-tmp/Free-Geometry
RR7=artifacts/diagnostics/final_protocol/7scenes
OUT=artifacts/diagnostics/final_protocol/da3_baseline
WS=workspace/da3_protocol_7scenes
LOG=$OUT/7scenes_loop.log
mkdir -p "$OUT"
echo "=== DA3 7scenes baseline start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/eval_da3_baseline.py --dataset 7scenes \
    --manifest "$RR7/scene_manifest.json" \
    --out "$OUT/7scenes_baseline.json" >> "$LOG" 2>&1 || { echo "BASELINE FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 7scenes TTA C2M start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/train_da3_protocol.py --dataset 7scenes \
    --scenes chess fire heads office pumpkin redkitchen stairs \
    --output_root "$WS" --steps 100 >> "$LOG" 2>&1 || { echo "TTA FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 7scenes loop DONE $(date '+%F %T') ===" >> "$LOG"
