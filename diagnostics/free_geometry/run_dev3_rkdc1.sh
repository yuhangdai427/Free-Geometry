#!/bin/bash
# dev3 W14：C2M_RKDC1（couple 权重 1.0，对照 W10 的 0.3）
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_dev3
LOG=$RR/stream_dev3_rkdc1.log

echo "=== train $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1 \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1

echo "=== eval infer $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1 --view_subsets 100v >> "$LOG" 2>&1

echo "=== metrics $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "A0_baseline@100v" "C2M_RKDC1@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1_dev3.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_dev3_rkdc1.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results
echo "=== DONE $(date '+%F %T') ===" >> "$LOG"
echo "DEV3_RKDC1 COMPLETE"
