#!/bin/bash
# 整夜主链：四数据集 × C2M_RKDC1（v3.0 最终形态）
# scannetpp20(v3 manifest, 7bc random) → 7scenes → hiroom → eth3d
# baseline 永不重跑（SKIP_BASELINE=1；run_eval 只评 TTA 臂）
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
LOG=artifacts/diagnostics/overnight_v3_chain.log
echo "=== overnight chain start $(date '+%F %T') ===" >> "$LOG"

run_one () {  # $1=ds $2=run_root $3=views(100v|allv)
  local DS=$1 RR=$2 VS=$3
  echo "=== [$DS] train $(date '+%F %T') ===" >> "$LOG"
  python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1 \
      --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
  echo "=== [$DS] eval infer $(date '+%F %T') ===" >> "$LOG"
  SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms C2M_RKDC1 --view_subsets "$VS" >> "$LOG" 2>&1
  echo "=== [$DS] metrics $(date '+%F %T') ===" >> "$LOG"
  python3 "$DG/run_eval.py" --run_root "$RR" --datas "$DS" \
      --experiments "C2M_RKDC1@$VS" >> "$LOG" 2>&1
  mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1.json"
  python3 "$DG/depth_metrics.py" --run_root "$RR" \
      --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1.csv" >> "$LOG" 2>&1
  rm -rf "$RR"/eval32/*/model_results
  echo "=== [$DS] DONE $(date '+%F %T') ===" >> "$LOG"
}

run_one scannetpp artifacts/diagnostics/final_protocol/scannetpp_v3 100v
run_one 7scenes artifacts/diagnostics/final_protocol/7scenes 100v
run_one hiroom artifacts/diagnostics/final_protocol/hiroom allv
run_one eth3d artifacts/diagnostics/final_protocol/eth3d allv
echo "=== overnight chain COMPLETE $(date '+%F %T') ===" >> "$LOG"
echo "OVERNIGHT_V3 COMPLETE"
