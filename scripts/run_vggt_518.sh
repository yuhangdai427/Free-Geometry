#!/bin/bash
# VGGT full rerun at native 518 resolution (not DA3's 504).
# Baseline + maskrel + RKDC1H, all-position loss, all 4 datasets.
# New run_root to preserve 504 results for comparison.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
DG=diagnostics/free_geometry
OUT=artifacts/diagnostics/final_protocol_518
PY=/root/miniconda3/envs/da3/bin/python

# Copy manifests (resolution-independent: frame indices only)
for ds in 7scenes eth3d hiroom scannetpp; do
  mkdir -p $OUT/$ds
  cp artifacts/diagnostics/final_protocol/$ds/scene_manifest.json $OUT/$ds/
done

run_ds () {
  local ds=$1
  local RR=$OUT/$ds
  local vc=$([ "$ds" = "7scenes" ] || [ "$ds" = "scannetpp" ] && echo "100v" || echo "allv")
  # Train both arms in one call
  echo "[518] $ds TRAIN START $(date '+%F %T')"
  $PY $DG/train_arms.py --run_root $RR \
      --arms C2M_maskrel C2M_RKDC1H --epochs 10 --seed 0 \
      --no_eval32 --loss_all_pos >> logs/vggt518_${ds}_train.log 2>&1 \
      || { echo "[518] $ds TRAIN FAILED"; return 1; }
  # Eval (includes a0 baseline automatically)
  echo "[518] $ds EVALVC START $(date '+%F %T')"
  $PY $DG/eval_viewcounts.py --manifest $RR/scene_manifest.json \
      --ckpt_root $RR/ckpts --run_root $RR --step 100 \
      --arms C2M_maskrel C2M_RKDC1H --view_subsets $vc >> logs/vggt518_${ds}_eval.log 2>&1 \
      || { echo "[518] $ds EVALVC FAILED"; return 1; }
  echo "[518] $ds RUNEVAL START $(date '+%F %T')"
  $PY $DG/run_eval.py --run_root $RR --datas $ds >> logs/vggt518_${ds}_eval.log 2>&1 \
      || { echo "[518] $ds RUNEVAL FAILED"; return 1; }
  mv $RR/eval32_metrics.json $RR/eval32_metrics_518.json
  echo "[518] $ds COMPLETE $(date '+%F %T')"
}

# Serial per dataset (train_arms internally serial per arm, 2 GPU lanes OK)
run_ds 7scenes
run_ds eth3d
run_ds hiroom
run_ds scannetpp
echo "[518] ALL DONE $(date '+%F %T')"
