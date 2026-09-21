#!/bin/bash
# DA3 16:4 + all-position + LR-fixed: clean ratio ablation vs existing 8:4 results.
# Everything identical except teacher_N=16 instead of 8.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

run () {
  local ds=$1 scenes=$2
  local out=workspace/da3_protocol_${ds}_t16s4_lossall
  mkdir -p "$out"
  echo "[164] $ds START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h \
      --teacher_N 16 --n_shared 4 --n_train 10 \
      --mask_ratio 0.5 --loss_all_pos \
      > logs/da3_${ds}_t16s4_lossall.log 2>&1 \
      || { echo "[164] $ds FAILED"; return 1; }
  echo "[164] $ds DONE $(date '+%F %T')"
}

# Lane 1: 7scenes then eth3d
run 7scenes "chess fire heads office pumpkin redkitchen stairs"
run eth3d "courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"
echo "[164-lane1] DONE $(date '+%F %T')"
