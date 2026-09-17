#!/bin/bash
# Lane B: Pi3 — waits for tonight's lossall chain (incl. orch3) to release the
# GPU lane, then runs baselines + TTA arms serially.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
RUNS=workspace/fgmig/runs

echo "[laneB] waiting for lossall chain + orch3 to finish $(date '+%F %T')"
while pgrep -f "run_lossall_8cells.sh|run_da3_spp_lossall" > /dev/null; do sleep 60; done
echo "[laneB] GPU lane free, starting Pi3 $(date '+%F %T')"

run () {
  local ds=$1 arm=$2
  local out=$RUNS/pi3_${ds}_${arm}
  [ -f "$out/metrics_attempted" ] && { echo "[laneB] skip $out"; return; }
  mkdir -p "$out"; touch "$out/metrics_attempted"
  echo "[laneB] pi3 $ds $arm START $(date '+%F %T')"
  $PY scripts/train_fg_protocol.py --model pi3 --dataset $ds --arm $arm \
      --output_root "$out" >> logs/fgmig_pi3_${ds}_${arm}.log 2>&1 \
      || { echo "[laneB] pi3 $ds $arm TRAIN-FAILED"; return 1; }
  echo "[laneB] pi3 $ds $arm EXPORT-DONE $(date '+%F %T')"
}

for ds in 7scenes eth3d hiroom scannetpp; do run $ds a0; done
for ds in 7scenes eth3d hiroom scannetpp; do run $ds rkdc_allpos; done
echo "[laneB] ALL DONE $(date '+%F %T')"
