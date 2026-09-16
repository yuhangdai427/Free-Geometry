#!/bin/bash
# 7scenes/stairs 单场景 × C2M_RKDC1H —— 验证 Huber 版 rkd 在 rkd 毒性最强场景上是否仍崩
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/7scenes_stairs_probe
LOG="$RR/chain.log"
echo "=== stairs RKDC1H start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1H \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
NCKPT=$(find "$RR/ckpts" -name "*RKDC1H*" | wc -l)
echo "RKDC1H ckpt entries: $NCKPT (expect stairs only)" >> "$LOG"
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1H --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas 7scenes \
    --experiments "C2M_RKDC1H@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1H.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1H.csv" >> "$LOG" 2>&1
echo "=== stairs RKDC1H COMPLETE $(date '+%F %T') ===" >> "$LOG"
