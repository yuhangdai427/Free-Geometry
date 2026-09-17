#!/bin/bash
# Catch-up: rerun the 4 dvlt a0 baselines that crashed on the student_label
# bug (fixed in adapters/dvlt.py). Waits for laneA's main dvlt queue to finish
# so we stay at 2 GPU lanes, then reruns (markers cleared by the launcher).
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYD=/root/autodl-tmp/conda/envs/dvlt/bin/python
export PYTHONPATH=/root/autodl-tmp/dvlt/src
RUNS=workspace/fgmig/runs

echo "[catchup] waiting for laneA dvlt queue $(date '+%F %T')"
while ! grep -q "ALL DONE" logs/fgmig_laneA.log 2>/dev/null; do sleep 60; done

for ds in 7scenes eth3d hiroom scannetpp; do
  out=$RUNS/dvlt_${ds}_a0
  rm -f "$out/metrics_attempted"
  mkdir -p "$out"; touch "$out/metrics_attempted"
  echo "[catchup] dvlt $ds a0 START $(date '+%F %T')"
  $PYD scripts/train_fg_protocol.py --model dvlt --dataset $ds --arm a0 \
      --output_root "$out" >> logs/fgmig_dvlt_${ds}_a0.log 2>&1 \
      && echo "[catchup] dvlt $ds a0 OK $(date '+%F %T')" \
      || echo "[catchup] dvlt $ds a0 FAILED"
done
echo "[catchup] DONE $(date '+%F %T')"
