#!/bin/bash
# relprobe: 3 F1-worst scannetpp scenes (21d970d8de/3e8bba0176/c5439f4607) x C2M_RKDC1R (RKDC1 + relpose addback)
# 2026-09-15, runs CONCURRENT with overnight chain (16v train ~24GB, chain peak ~36GB, total < 97GB)
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_relprobe
LOG="$RR/relprobe_chain.log"
echo "=== relprobe start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1R \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1R --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --eval_dirs "$RR/eval32" >> "$LOG" 2>&1
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1R.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results 2>/dev/null
echo "=== relprobe COMPLETE $(date '+%F %T') ===" >> "$LOG"
