#!/bin/bash
# DVLT two-stage LoRA (B+): rank 32 per stage, split at loop 6, 1.6M params.
# Compare against shared LoRA (0.8M) — tests "do early/late loops need
# different adaptations?"
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYD=/root/autodl-tmp/conda/envs/dvlt/bin/python
PY=/root/miniconda3/envs/da3/bin/python
export PYTHONPATH=/root/autodl-tmp/dvlt/src

for ds in 7scenes eth3d hiroom scannetpp; do
  out=workspace/fgmig/runs/dvlt_${ds}_rkdc_2stage
  mkdir -p "$out"
  echo "[2stage] dvlt $ds START $(date '+%F %T')"
  $PYD scripts/train_fg_protocol.py --model dvlt --dataset $ds \
      --arm rkdc_allpos --lora_variant two_stage \
      --output_root "$out" >> logs/fgmig_dvlt_${ds}_2stage.log 2>&1 \
      || { echo "[2stage] dvlt $ds FAILED"; continue; }
  $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
      >> logs/fgmig_dvlt_${ds}_2stage.log 2>&1 \
      || echo "[2stage] metrics $ds FAILED"
  echo "[2stage] dvlt $ds DONE $(date '+%F %T')"
done
echo "[2stage] ALL DONE $(date '+%F %T')"
