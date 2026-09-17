#!/bin/bash
# VGGT-Omega maskrel_allpos arm (maskdistill all-position + rel pose), 4 datasets.
# Tests whether the VGGT-family maskrel>rkd pattern transfers to Omega.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

for ds in 7scenes eth3d hiroom scannetpp; do
  out=workspace/fgmig/runs/omega_${ds}_maskrel_allpos
  mkdir -p "$out"
  echo "[maskrel] omega $ds START $(date '+%F %T')"
  $PY scripts/train_fg_protocol.py --model omega --dataset $ds --arm maskrel_allpos \
      --output_root "$out" >> logs/fgmig_omega_${ds}_maskrel.log 2>&1 \
      || { echo "[maskrel] omega $ds TRAIN-FAILED"; continue; }
  $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
      >> logs/fgmig_omega_${ds}_maskrel.log 2>&1 || echo "[maskrel] metrics $ds FAILED"
  echo "[maskrel] omega $ds DONE $(date '+%F %T')"
done
echo "[maskrel] ALL DONE $(date '+%F %T')"
