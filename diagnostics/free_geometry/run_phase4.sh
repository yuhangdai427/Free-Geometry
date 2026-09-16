#!/bin/bash
# Phase-4 sub8 gating: stacked arms + student-selection control.
# Run from repo root: bash diagnostics/free_geometry/run_phase4.sh
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
P4=artifacts/diagnostics/final_protocol_phase4
CTL=artifacts/diagnostics/final_protocol_studentctl

views() { case "$1" in scannetpp|7scenes) echo "100v";; *) echo "allv";; esac }

run_arm () {  # $1=root $2=ds $3=arm
  local RR="$1/$2" ARM="$3"
  local LOG="$RR/stream_$ARM.log"
  echo "=== [$2] $ARM train $(date '+%F %T') ===" >> "$LOG"
  python "$DG/train_arms.py" --run_root "$RR" --arms "$ARM" \
      --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
  echo "=== [$2] $ARM eval $(date '+%F %T') ===" >> "$LOG"
  python "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms "$ARM" --view_subsets "$(views "$2")" >> "$LOG" 2>&1
  python "$DG/run_eval.py" --run_root "$RR" --datas "$2" >> "$LOG" 2>&1
  python "$DG/depth_metrics.py" --run_root "$RR" \
      --manifest "$RR/scene_manifest.json" >> "$LOG" 2>&1
  rm -rf "$RR"/eval32/*/model_results
  echo "=== [$2] $ARM DONE $(date '+%F %T') ===" >> "$LOG"
}

# stream 1 (args: scannetpp 7scenes): stacked arms on main sub8 manifests
# stream 2 (args: hiroom eth3d)
if [ "${PHASE4_STREAM:-1}" = "1" ]; then DS_LIST="scannetpp 7scenes"; else DS_LIST="hiroom eth3d"; fi
for ds in $DS_LIST; do
  for arm in C2M_CamRel CONFD_REL C2M_CTM; do
    run_arm "$P4" "$ds" "$arm"
  done
  run_arm "$CTL" "$ds" "C2M_maskrel"   # student clustered-control
done
echo "PHASE4 STREAM ${PHASE4_STREAM:-1} COMPLETE"
