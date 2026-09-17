#!/bin/bash
# lrfix round 2: DVLT hiroom + scannetpp at lr 1e-6 (round 1 flipped 7s/eth3d
# from strongly negative to double-positive).
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYD=/root/autodl-tmp/conda/envs/dvlt/bin/python
PY=/root/miniconda3/envs/da3/bin/python
export PYTHONPATH=/root/autodl-tmp/dvlt/src

for ds in hiroom scannetpp; do
  out=workspace/fgmig/runs/dvlt_${ds}_rkdc_lr1e6
  mkdir -p "$out"
  echo "[lrfix2] dvlt $ds lr1e-6 START $(date '+%F %T')"
  $PYD scripts/train_fg_protocol.py --model dvlt --dataset $ds --arm rkdc_allpos \
      --lr 1e-6 --output_root "$out" >> logs/fgmig_dvlt_${ds}_lr1e6.log 2>&1 \
      || { echo "[lrfix2] dvlt $ds FAILED"; continue; }
  $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
      >> logs/fgmig_dvlt_${ds}_lr1e6.log 2>&1 || echo "[lrfix2] metrics $ds FAILED"
  echo "[lrfix2] dvlt $ds lr1e-6 DONE $(date '+%F %T')"
done
echo "[lrfix2] ALL DONE $(date '+%F %T')"
