#!/bin/bash
# DA3 hiroom 闭环: baseline(无适配) -> TTA rkdc1h（主臂，已验证）
set -u
cd /root/autodl-tmp/Free-Geometry
RRH=artifacts/diagnostics/final_protocol/hiroom
OUT=artifacts/diagnostics/final_protocol/da3_baseline
WS=workspace/da3_protocol_hiroom_rkdc1h
LOG=$OUT/hiroom_loop.log
mkdir -p "$OUT"
SCENES=$(python3 -c "import json; print(' '.join(sorted(json.load(open('$RRH/scene_manifest.json'))['scenes'])))")
if [ -s "$OUT/hiroom_baseline.json" ]; then
    echo "=== hiroom baseline exists, skip $(date '+%F %T') ===" >> "$LOG"
else
    echo "=== DA3 hiroom baseline start $(date '+%F %T') ===" >> "$LOG"
    python3 scripts/eval_da3_baseline.py --dataset hiroom \
        --manifest "$RRH/scene_manifest.json" \
        --out "$OUT/hiroom_baseline.json" >> "$LOG" 2>&1 || { echo "BASELINE FAILED" >> "$LOG"; exit 1; }
fi
echo "=== DA3 hiroom rkdc1h start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/train_da3_protocol.py --dataset hiroom --scenes $SCENES \
    --output_root "$WS" --steps 100 --arm rkdc1h >> "$LOG" 2>&1 || { echo "TTA FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 hiroom loop DONE $(date '+%F %T') ===" >> "$LOG"
