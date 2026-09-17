#!/bin/bash
# DVLT LoRA rerun: rank-32 LoRA on the shared recurrent block (same hyperparams
# as omega/pi3), replacing the full-FT runs. All 4 datasets, a0 + rkdc_allpos.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYD=/root/autodl-tmp/conda/envs/dvlt/bin/python
PY=/root/miniconda3/envs/da3/bin/python
export PYTHONPATH=/root/autodl-tmp/dvlt/src

for ds in 7scenes eth3d hiroom scannetpp; do
  for arm in a0 rkdc_allpos; do
    out=workspace/fgmig/runs/dvlt_${ds}_${arm}_lora
    mkdir -p "$out"
    echo "[dvlt-lora] $ds $arm START $(date '+%F %T')"
    $PYD scripts/train_fg_protocol.py --model dvlt --dataset $ds --arm $arm \
        --output_root "$out" >> logs/fgmig_dvlt_${ds}_${arm}_lora.log 2>&1 \
        || { echo "[dvlt-lora] $ds $arm FAILED"; continue; }
    echo "[dvlt-lora] $ds $arm EXPORT $(date '+%F %T')"
    $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
        >> logs/fgmig_dvlt_${ds}_${arm}_lora.log 2>&1 \
        || echo "[dvlt-lora] metrics $ds FAILED"
    echo "[dvlt-lora] $ds $arm DONE $(date '+%F %T')"
  done
done
echo "[dvlt-lora] ALL DONE $(date '+%F %T')"
