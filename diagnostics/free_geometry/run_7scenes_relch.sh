#!/bin/bash
# 7scenes 全量 × C2M_RELCH（rel + rkd_huber + couple），补全 RELCH 四数据集矩阵
# 与今日 RELC/RKDC1 同 manifest（16:4, 100v），直接可比
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/7scenes
LOG="$RR/relch_chain.log"
echo "=== 7scenes RELCH start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RELCH \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
NCKPT=$(ls "$RR/ckpts" | grep -c RELCH || true)
echo "RELCH ckpts: $NCKPT (expect 7 scenes)" >> "$LOG"
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RELCH --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas 7scenes \
    --experiments "C2M_RELCH@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RELCH.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RELCH.csv" >> "$LOG" 2>&1
echo "=== 7scenes RELCH COMPLETE $(date '+%F %T') ===" >> "$LOG"
