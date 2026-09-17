#!/bin/bash
# Fix-experiment: DVLT full-FT at lr 1e-6 (the lr 1e-5 recipe hurt all 4 cells;
# losses dropped up to 8x -> updates too aggressive). 7scenes + eth3d first.
# Waits for laneB (pi3) to release the GPU lane. Metrics computed in-window.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYD=/root/autodl-tmp/conda/envs/dvlt/bin/python
PY=/root/miniconda3/envs/da3/bin/python
export PYTHONPATH=/root/autodl-tmp/dvlt/src

echo "[lrfix] waiting for laneB $(date '+%F %T')"
while ! grep -q "ALL DONE" logs/fgmig_laneB.log 2>/dev/null; do sleep 60; done

for ds in 7scenes eth3d; do
  out=workspace/fgmig/runs/dvlt_${ds}_rkdc_lr1e6
  mkdir -p "$out"
  echo "[lrfix] dvlt $ds lr1e-6 START $(date '+%F %T')"
  $PYD scripts/train_fg_protocol.py --model dvlt --dataset $ds --arm rkdc_allpos \
      --lr 1e-6 --output_root "$out" >> logs/fgmig_dvlt_${ds}_lr1e6.log 2>&1 \
      || { echo "[lrfix] dvlt $ds FAILED"; continue; }
  $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
      >> logs/fgmig_dvlt_${ds}_lr1e6.log 2>&1 || echo "[lrfix] metrics $ds FAILED"
  echo "[lrfix] dvlt $ds lr1e-6 DONE $(date '+%F %T')"
done
echo "[lrfix] ALL DONE $(date '+%F %T')"
