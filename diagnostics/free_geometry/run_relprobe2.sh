#!/bin/bash
# relprobe2: 3 F1-worst scannetpp scenes — RKDC1R 重推理(修复 run_eval 参数 + npz 被误删) + RKDC1A(abs_raw 加回) 新训练
# 2026-09-15. 与主链并发：inference ~35GB + 主链训练 ~24GB < 97GB
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_relprobe
LOG="$RR/relprobe2_chain.log"
echo "=== relprobe2 start $(date '+%F %T') ===" >> "$LOG"
# 1) 训练 RKDC1A（RKDC1R 的 ckpts 已存在，train_arms 会跳过或重训；这里只训新臂）
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1A \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
# 2) 双臂推理（RKDC1R 的 npz 被误删需重生；RKDC1A 是新的）
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1R C2M_RKDC1A --view_subsets 100v >> "$LOG" 2>&1
# 3) 融合+位姿指标（正确参数 --experiments）
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "C2M_RKDC1R@100v" "C2M_RKDC1A@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1RA.json"
# 4) RKDC1A 深度指标（RKDC1R 的已在 depth_metrics_RKDC1R.csv）
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1A.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results 2>/dev/null
echo "=== relprobe2 COMPLETE $(date '+%F %T') ===" >> "$LOG"
