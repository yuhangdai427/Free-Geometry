#!/bin/bash
# Lane C (CPU only): poll for completed migration runs and compute metrics.
# A run is eligible when summary.json exists; metrics.json marks completion.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
RUNS=workspace/fgmig/runs
EXPECTED=${1:-25}

for i in $(seq 1 420); do  # up to 10.5 h
  done_n=$(ls $RUNS/*/metrics.json 2>/dev/null | wc -l)
  if [ "$done_n" -ge "$EXPECTED" ]; then echo "[laneC] all $EXPECTED metrics done"; break; fi
  for out in $RUNS/*/; do
    out=${out%/}
    [ -f "$out/summary.json" ] && [ ! -f "$out/metrics.json" ] && [ ! -f "$out/metrics_running" ] || continue
    ds=$(basename "$out" | cut -d_ -f2)
    touch "$out/metrics_running"
    echo "[laneC] metrics for $(basename $out) $(date '+%F %T')"
    timeout 7200 $PY scripts/fg_eval_from_npz.py --dataset "$ds" --output_root "$out" \
      > "logs/fgmig_metrics_$(basename $out).log" 2>&1 \
      && rm -f "$out/metrics_running" \
      || { echo "[laneC] metrics FAILED for $out"; rm -f "$out/metrics_running"; touch "$out/metrics_failed"; }
  done
  sleep 90
done
echo "[laneC] EXIT $(date '+%F %T')"
