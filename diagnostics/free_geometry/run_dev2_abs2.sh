#!/bin/bash
# dev2 ABS 2x2 补格：C2M_ABSW1(归一化,w=1) + C2M_ABSR5(原始靶,w=5)
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_dev2
LOG=$RR/stream_dev2_abs22.log

echo "=== train $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_ABSW1 C2M_ABSR5 \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1

echo "=== eval infer $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_ABSW1 C2M_ABSR5 --view_subsets 100v >> "$LOG" 2>&1

echo "=== metrics $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "A0_baseline@100v" "C2M_ABS@100v" "C2M_ABSR@100v" "C2M_ABSW1@100v" "C2M_ABSR5@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_ABS_2x2.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_dev2_2x2.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results
echo "=== DONE $(date '+%F %T') ===" >> "$LOG"
echo "DEV2_2X2 COMPLETE"
