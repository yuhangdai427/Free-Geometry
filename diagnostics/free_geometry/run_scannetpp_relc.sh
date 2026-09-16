#!/bin/bash
# scannetpp_v3 全量 20 场景 × C2M_RELC（RELC 全量验证：无 rkd 时 scannetpp AUC/F1 表现（统一 loss 候选））
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_v3
LOG="$RR/relc_chain.log"
echo "=== scannetpp RKDC1H full20 start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RELC \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RELC --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "C2M_RELC@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RELC.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RELC.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/C2M_RELC@100v/model_results 2>/dev/null
echo "=== scannetpp RKDC1H full20 COMPLETE $(date '+%F %T') ===" >> "$LOG"
