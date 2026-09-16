#!/bin/bash
# 7scenes 全量 × C2M_RKLH（统一候选: feat+rel+1.5*rkd_local_huber+1.0*couple）
# 与今日 RKDC1 同 manifest（16:4），直接可比
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/7scenes
LOG="$RR/rklh_chain.log"
echo "=== 7scenes RKDC1R start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKLH \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKLH --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas 7scenes \
    --experiments "C2M_RKLH@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKLH.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKLH.csv" >> "$LOG" 2>&1
echo "=== 7scenes RKDC1R COMPLETE $(date '+%F %T') ===" >> "$LOG"
