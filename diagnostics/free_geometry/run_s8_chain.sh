#!/bin/bash
# Dynamic-student (s8/s2) chain: smoke -> full runs, all sequential.
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
S8=artifacts/diagnostics/final_protocol_s8
LOG=$S8/chain.log
mkdir -p $S8

echo "=== smoke 7scenes_s8t32 chess 20 steps $(date '+%F %T') ===" >> "$LOG"
python "$DG/train_arms.py" --run_root "$S8/7scenes_s8t32" --arms C2M_maskrel \
    --epochs 10 --seed 0 --no_eval32 --no_ckpt --scenes chess --max_steps 20 >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "SMOKE FAILED - abort" >> "$LOG"; exit 1; fi
echo "=== smoke OK $(date '+%F %T') ===" >> "$LOG"

for cfg in "7scenes_s8t32 7scenes 100v" "scannetpp_s8t24 scannetpp 100v" "hiroom_s2t8 hiroom allv" "hiroom_s2t4 hiroom allv"; do
  set -- $cfg
  V=$1; DS=$2; VS=$3
  RR=$S8/$V
  echo "=== [$V] train $(date '+%F %T') ===" >> "$LOG"
  python "$DG/train_arms.py" --run_root "$RR" --arms C2M_maskrel \
      --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "[$V] TRAIN FAILED" >> "$LOG"; continue; }
  echo "=== [$V] eval $(date '+%F %T') ===" >> "$LOG"
  python "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms C2M_maskrel --view_subsets "$VS" >> "$LOG" 2>&1 || { echo "[$V] EVAL FAILED" >> "$LOG"; continue; }
  python "$DG/run_eval.py" --run_root "$RR" --datas "$DS" >> "$LOG" 2>&1
  python "$DG/depth_metrics.py" --run_root "$RR" --manifest "$RR/scene_manifest.json" >> "$LOG" 2>&1
  rm -rf "$RR"/eval32/*/model_results
  echo "=== [$V] DONE $(date '+%F %T') ===" >> "$LOG"
done
echo "S8 CHAIN COMPLETE" >> "$LOG"
