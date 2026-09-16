#!/bin/bash
# huberprobe: 位姿过推型 3 场景(09c1414f1b/c5439f4607/578511c8a9) × C2M_RKDC1H (rkd Huber δ=0.2)
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_huberprobe
LOG="$RR/huberprobe_chain.log"
echo "=== huberprobe start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1H \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1H --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "C2M_RKDC1H@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1H.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1H.csv" >> "$LOG" 2>&1
echo "=== huberprobe COMPLETE $(date '+%F %T') ===" >> "$LOG"
