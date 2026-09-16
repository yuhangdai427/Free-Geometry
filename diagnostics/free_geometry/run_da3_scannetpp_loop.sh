#!/bin/bash
# DA3 scannetpp 闭环: baseline(无适配) -> TTA rkdc1h（主臂主场）
set -u
cd /root/autodl-tmp/Free-Geometry
RRS=artifacts/diagnostics/final_protocol/scannetpp_v3
OUT=artifacts/diagnostics/final_protocol/da3_baseline
WS=workspace/da3_protocol_scannetpp_rkdc1h
LOG=$OUT/scannetpp_loop.log
mkdir -p "$OUT"
SCENES=$(python3 -c "import json; print(' '.join(sorted(json.load(open('$RRS/scene_manifest.json'))['scenes'])))")
if [ -s "$OUT/scannetpp_baseline.json" ]; then
    echo "=== scannetpp baseline exists, skip $(date '+%F %T') ===" >> "$LOG"
else
    echo "=== DA3 scannetpp baseline start $(date '+%F %T') ===" >> "$LOG"
    python3 scripts/eval_da3_baseline.py --dataset scannetpp \
        --manifest "$RRS/scene_manifest.json" \
        --out "$OUT/scannetpp_baseline.json" >> "$LOG" 2>&1 || { echo "BASELINE FAILED" >> "$LOG"; exit 1; }
fi
echo "=== DA3 scannetpp rkdc1h start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/train_da3_protocol.py --dataset scannetpp --scenes $SCENES \
    --output_root "$WS" --steps 100 --arm rkdc1h >> "$LOG" 2>&1 || { echo "TTA FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 scannetpp loop DONE $(date '+%F %T') ===" >> "$LOG"
