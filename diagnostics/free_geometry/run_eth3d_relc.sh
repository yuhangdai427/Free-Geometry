#!/bin/bash
# 7scenes 全量 × C2M_RELC（鉴别: maskdistill+rel+couple(无rkd)，拆分7scenes毒性来源）
# 与今日 RKDC1 同 manifest（16:4），直接可比
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/eth3d
LOG="$RR/relc_eth3d_chain.log"
echo "=== 7scenes RKDC1R start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RELC \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RELC --view_subsets allv >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas eth3d \
    --experiments "C2M_RELC@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RELC.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RELC.csv" >> "$LOG" 2>&1
echo "=== 7scenes RKDC1R COMPLETE $(date '+%F %T') ===" >> "$LOG"
