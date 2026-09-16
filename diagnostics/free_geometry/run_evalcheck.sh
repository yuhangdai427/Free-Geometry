#!/bin/bash
# evalcheck: 用今天的 eval 代码复评 v2 时代的 16:4 maskrel chess ckpt
# 预期: 若 eval 一致 → AUC≈0.1856 (09-13 存档值); 若偏差大 → eval 代码变了
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/evalcheck_chess
LOG="$RR/chain.log"
echo "=== evalcheck start $(date '+%F %T') ===" >> "$LOG"
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root artifacts/diagnostics/final_protocol/7scenes/ckpts --run_root "$RR" --step 100 \
    --arms C2M_maskrel --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas 7scenes \
    --experiments "C2M_maskrel@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_maskrel_today.json"
echo "=== evalcheck COMPLETE $(date '+%F %T') ===" >> "$LOG"
