#!/bin/bash
# VGGT forced 8:4 (teacher_N=8) fill-in runs: per-dataset champion loss.
# eth3d/hiroom/7scenes -> C2M_maskrel, scannetpp -> C2M_RKDC1H.
# Serial, single lane; eval = 100v for N>=100 datasets, allv otherwise.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
OUT=artifacts/diagnostics/final_protocol_t8

run_ds () {
  local ds=$1 arm=$2 vc=$3
  local RR=$OUT/$ds
  echo "=== [$ds] arm=$arm vc=$vc TRAIN start $(date '+%F %T') ==="
  $PY $DG/train_arms.py --run_root $RR --arms $arm --epochs 10 --seed 0 --no_eval32 \
    || { echo "[$ds] TRAIN FAILED"; return 1; }
  echo "=== [$ds] EVALVC start $(date '+%F %T') ==="
  $PY $DG/eval_viewcounts.py --manifest $RR/scene_manifest.json \
    --ckpt_root $RR/ckpts --run_root $RR --step 100 --arms $arm --view_subsets $vc \
    || { echo "[$ds] EVALVC FAILED"; return 1; }
  echo "=== [$ds] RUNEVAL start $(date '+%F %T') ==="
  $PY $DG/run_eval.py --run_root $RR --datas $ds \
    || { echo "[$ds] RUNEVAL FAILED"; return 1; }
  mv $RR/eval32_metrics.json $RR/eval32_metrics_${arm}.json
  echo "=== [$ds] COMPLETE $(date '+%F %T') ==="
}

run_ds eth3d    C2M_maskrel allv
run_ds hiroom   C2M_maskrel allv
run_ds 7scenes  C2M_maskrel 100v
run_ds scannetpp C2M_RKDC1H 100v
echo "ALL DONE $(date '+%F %T')"
