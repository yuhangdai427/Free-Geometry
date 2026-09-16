#!/bin/bash
# Variant pipeline (teacher-N ablation). Usage: bash run_variant.sh <variant> <ds> <views>
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
V=$1; DS=$2; VS=$3
ARM=${ARM:-C2M_maskrel}
RR=${ROOT:-artifacts/diagnostics/final_protocol_variants}/$V
LOG=$RR/stream_$ARM.log
mkdir -p "$RR"
echo "=== [$V] $ARM train $(date '+%F %T') ===" >> "$LOG"
python "$DG/train_arms.py" --run_root "$RR" --arms "$ARM" \
    --epochs ${EPOCHS:-10} --seed 0 --no_eval32 >> "$LOG" 2>&1
echo "=== [$V] $ARM eval infer $(date '+%F %T') ===" >> "$LOG"
python "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms "$ARM" --view_subsets "$VS" >> "$LOG" 2>&1
echo "=== [$V] $ARM metrics $(date '+%F %T') ===" >> "$LOG"
python "$DG/run_eval.py" --run_root "$RR" --datas "$DS" --experiments "A0_baseline@$( [ "$VS" = "allv" ] && echo allv || echo 100v)" "$ARM@$( [ "$VS" = "allv" ] && echo allv || echo 100v)" >> "$LOG" 2>&1
  mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_$ARM.json"
python "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_$ARM.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results
echo "=== [$V] DONE $(date '+%F %T') ===" >> "$LOG"
echo "VARIANT $V COMPLETE"
