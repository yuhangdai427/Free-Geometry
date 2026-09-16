#!/bin/bash
# eth3d 全量 × C2M_RELCH（rel + rkd_huber + couple），补全 RELCH 四数据集矩阵
# 与 RKDC1H 同 manifest（allv 评测），直接可比
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/eth3d
LOG="$RR/relch_chain.log"
echo "=== eth3d RELCH start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RELCH \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
NCKPT=$(ls "$RR/ckpts" | grep -c RELCH || true)
echo "RELCH ckpts: $NCKPT (expect 11 scenes)" >> "$LOG"
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RELCH --view_subsets allv >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas eth3d \
    --experiments "C2M_RELCH@allv" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RELCH.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RELCH.csv" >> "$LOG" 2>&1
echo "=== eth3d RELCH COMPLETE $(date '+%F %T') ===" >> "$LOG"
