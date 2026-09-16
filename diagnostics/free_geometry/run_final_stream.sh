#!/bin/bash
# Final-protocol per-dataset pipeline: train -> eval infer -> metrics -> cleanup.
# Usage (from repo root): ARM=C2M_maskrel bash diagnostics/free_geometry/run_final_stream.sh scannetpp 7scenes
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
ARM=${ARM:-C2M_maskrel}
for ds in "$@"; do
  RR=artifacts/diagnostics/final_protocol/$ds
  LOG=$RR/stream_$ARM.log
  mkdir -p "$RR"
  echo "=== [$ds] $ARM train $(date '+%F %T') ===" >> "$LOG"
  python "$DG/train_arms.py" --run_root "$RR" --arms "$ARM" \
      --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1
  echo "=== [$ds] $ARM eval infer $(date '+%F %T') ===" >> "$LOG"
  if [ "$ds" = "scannetpp" ] || [ "$ds" = "7scenes" ]; then VS="100v 8v 4v"; else VS="allv"; fi
  python "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms "$ARM" --view_subsets $VS >> "$LOG" 2>&1
  echo "=== [$ds] $ARM metrics $(date '+%F %T') ===" >> "$LOG"
  python "$DG/run_eval.py" --run_root "$RR" --datas "$ds" >> "$LOG" 2>&1
  python "$DG/depth_metrics.py" --run_root "$RR" \
      --manifest "$RR/scene_manifest.json" >> "$LOG" 2>&1
  rm -rf "$RR"/eval32/*/model_results
  echo "=== [$ds] $ARM DONE $(date '+%F %T') ===" >> "$LOG"
done
echo "STREAM COMPLETE: $ARM $*"
