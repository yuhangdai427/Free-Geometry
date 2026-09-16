#!/bin/bash
# relprobe3: 续跑 relprobe2（被 600s 默认超时杀掉）——补 RKDC1A 推理 + 双臂融合评测
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_relprobe
LOG="$RR/relprobe3_chain.log"
echo "=== relprobe3 start $(date '+%F %T') ===" >> "$LOG"
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1A --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "C2M_RKDC1R@100v" "C2M_RKDC1A@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1RA.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1A.csv" >> "$LOG" 2>&1
echo "=== relprobe3 COMPLETE $(date '+%F %T') ===" >> "$LOG"
