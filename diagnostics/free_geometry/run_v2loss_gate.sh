#!/bin/bash
# v2 loss gating on sub8 (old selection, phase4 manifests):
#   B5_CTK (maskdistill+CamTokHC) on scannetpp/hiroom/eth3d
#   C2M_CTK (maskdistill+CamTokHC+capped rel-pose) on 7scenes
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
P4=artifacts/diagnostics/final_protocol_phase4
run () { # $1=ds $2=arm $3=views
  local RR=$P4/$1
  local LOG=$RR/stream_$2.log
  echo "=== [$1] $2 train $(date '+%F %T') ===" >> "$LOG"
  python "$DG/train_arms.py" --run_root "$RR" --arms "$2" \
      --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
  python "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms "$2" --view_subsets "$3" >> "$LOG" 2>&1
  python "$DG/run_eval.py" --run_root "$RR" --datas "$1" >> "$LOG" 2>&1
  python "$DG/depth_metrics.py" --run_root "$RR" \
      --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_$2.csv" >> "$LOG" 2>&1
  mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_$2.json"
  rm -rf "$RR"/eval32/*/model_results
  echo "=== [$1] $2 DONE $(date '+%F %T') ===" >> "$LOG"
}
if [ "${GATE_STREAM:-1}" = "1" ]; then
  run scannetpp B5_CTK 100v
  run 7scenes C2M_CTK 100v
else
  run hiroom B5_CTK allv
  run eth3d B5_CTK allv
fi
echo "GATE STREAM ${GATE_STREAM:-1} COMPLETE"
