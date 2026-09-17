#!/bin/bash
# Pi3 rerun with the pointmap-K fix (model-resolution K estimated from Pi3's own
# local pointmaps; previous GT-K was native-res -> 12x focal error on eth3d).
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

for ds in 7scenes eth3d hiroom scannetpp; do
  out=workspace/fgmig/runs/pi3_${ds}_a0
  rm -f "$out/metrics_attempted"; mkdir -p "$out"; touch "$out/metrics_attempted"
  echo "[kfix] pi3 $ds a0 START $(date '+%F %T')"
  $PY scripts/train_fg_protocol.py --model pi3 --dataset $ds --arm a0 \
      --output_root "$out" >> logs/fgmig_pi3_${ds}_a0_kfix.log 2>&1 \
      || { echo "[kfix] pi3 $ds a0 FAILED"; continue; }
  $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
      >> logs/fgmig_pi3_${ds}_a0_kfix.log 2>&1 || echo "[kfix] metrics $ds FAILED"
  echo "[kfix] pi3 $ds a0 DONE $(date '+%F %T')"
done
for ds in 7scenes eth3d hiroom scannetpp; do
  out=workspace/fgmig/runs/pi3_${ds}_rkdc_allpos
  rm -f "$out/metrics_attempted"; mkdir -p "$out"; touch "$out/metrics_attempted"
  echo "[kfix] pi3 $ds rkdc START $(date '+%F %T')"
  $PY scripts/train_fg_protocol.py --model pi3 --dataset $ds --arm rkdc_allpos \
      --output_root "$out" >> logs/fgmig_pi3_${ds}_rkdc_kfix.log 2>&1 \
      || { echo "[kfix] pi3 $ds rkdc FAILED"; continue; }
  $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
      >> logs/fgmig_pi3_${ds}_rkdc_kfix.log 2>&1 || echo "[kfix] metrics $ds FAILED"
  echo "[kfix] pi3 $ds rkdc DONE $(date '+%F %T')"
done
echo "[kfix] ALL DONE $(date '+%F %T')"
