#!/bin/bash
# 7scenes_long: teacher32:student8 等距配对 × C2M_RKDC1（治 16:4 在长稠密序列上 AUC 崩盘）
# 2026-09-15. 等 relprobe2 结束后启动（GPU 串行纪律）
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/7scenes_long
LOG="$RR/chain.log"
echo "=== 7scenes_long start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1 \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1 --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas 7scenes \
    --experiments "C2M_RKDC1@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results 2>/dev/null
echo "=== 7scenes_long COMPLETE $(date '+%F %T') ===" >> "$LOG"
