#!/bin/bash
# Watcher: when a DA3 ndispatch dataset finishes, auto-run its 10-frame re-eval.
# Usage: bash scripts/watch_da3_eval10.sh <dataset>   (e.g. eth3d)
set -u
cd /root/autodl-tmp/Free-Geometry
ds=$1
out=workspace/ndispatch/da3_$ds
log=logs/ndispatch_da3_${ds}_eval10.log
echo "[watch] waiting for $out/smoke_summary.json ..."
while [ ! -f "$out/smoke_summary.json" ]; do sleep 60; done
# small grace: summary is written at the very end of the run
sleep 120
while pgrep -f "output_root workspace/ndispatch/da3_$ds " >/dev/null 2>&1; do sleep 60; done
echo "[watch] $ds done, starting eval10 $(date '+%F %T')"
PYTORCH_ALLOC_CONF=expandable_segments:True \
  /root/miniconda3/envs/da3/bin/python scripts/eval_da3_10frames.py --dataset $ds \
  > "$log" 2>&1 && echo "[watch] $ds eval10 DONE $(date '+%F %T')" \
  || echo "[watch] $ds eval10 FAILED — see $log"
