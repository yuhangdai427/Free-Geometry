#!/bin/bash
# DVLT LR follow-up: lr 3e-7 on the two open cells (hiroom AUC-neutral,
# scannetpp negative), plus an m_allpos component control on scannetpp at 1e-6
# (is the damage from the geometry terms or the distillation itself?).
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYD=/root/autodl-tmp/conda/envs/dvlt/bin/python
PY=/root/miniconda3/envs/da3/bin/python
export PYTHONPATH=/root/autodl-tmp/dvlt/src

run () {
  local ds=$1 arm=$2 lr=$3
  local out=workspace/fgmig/runs/dvlt_${ds}_${arm}_lr${lr}
  mkdir -p "$out"
  echo "[lr3e7] dvlt $ds $arm lr=$lr START $(date '+%F %T')"
  $PYD scripts/train_fg_protocol.py --model dvlt --dataset $ds --arm $arm \
      --lr $lr --output_root "$out" >> logs/fgmig_dvlt_${ds}_${arm}_lr${lr}.log 2>&1 \
      || { echo "[lr3e7] dvlt $ds $arm FAILED"; return 1; }
  $PY scripts/fg_eval_from_npz.py --dataset $ds --output_root "$out" \
      >> logs/fgmig_dvlt_${ds}_${arm}_lr${lr}.log 2>&1 || echo "[lr3e7] metrics $ds FAILED"
  echo "[lr3e7] dvlt $ds $arm lr=$lr DONE $(date '+%F %T')"
}

run scannetpp rkdc_allpos 3e-7
run hiroom    rkdc_allpos 3e-7
run scannetpp m_allpos    1e-6
echo "[lr3e7] ALL DONE $(date '+%F %T')"
