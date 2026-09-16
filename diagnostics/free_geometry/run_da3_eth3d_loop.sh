#!/bin/bash
# DA3 eth3d 闭环: baseline(无适配) -> TTA C2M(maskrel) + pw0(纯maskdistill) 双臂
set -u
cd /root/autodl-tmp/Free-Geometry
RRE=artifacts/diagnostics/final_protocol/eth3d
OUT=artifacts/diagnostics/final_protocol/da3_baseline
WS=workspace/da3_protocol_eth3d_v2
LOG=$OUT/eth3d_loop.log
mkdir -p "$OUT"
SCENES=$(python3 -c "import json; print(' '.join(sorted(json.load(open('$RRE/scene_manifest.json'))['scenes'])))")
if [ -s "$OUT/eth3d_baseline.json" ]; then
    echo "=== baseline exists, skip (永不重跑) $(date '+%F %T') ===" >> "$LOG"
else
    echo "=== DA3 eth3d baseline start $(date '+%F %T') scenes: $SCENES ===" >> "$LOG"
    python3 scripts/eval_da3_baseline.py --dataset eth3d \
        --manifest "$RRE/scene_manifest.json" \
        --out "$OUT/eth3d_baseline.json" >> "$LOG" 2>&1 || { echo "BASELINE FAILED" >> "$LOG"; exit 1; }
fi
echo "=== DA3 eth3d TTA C2M start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/train_da3_protocol.py --dataset eth3d --scenes $SCENES \
    --output_root "$WS" --steps 100 >> "$LOG" 2>&1 || { echo "TTA C2M FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 eth3d TTA pw0 start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/train_da3_protocol.py --dataset eth3d --scenes $SCENES \
    --output_root "${WS}_pw0" --steps 100 --pose_weight 0.0 >> "$LOG" 2>&1 || { echo "TTA pw0 FAILED" >> "$LOG"; exit 1; }
echo "=== DA3 eth3d loop DONE $(date '+%F %T') ===" >> "$LOG"
