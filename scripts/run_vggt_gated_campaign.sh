#!/bin/bash
# VGGT gated rawrel (pose_gate now applied in C2M_rawrel): train + eval for one
# dataset. Usage: run_vggt_gated_campaign.sh dtu|dtu64
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
DS=$1
MAN=workspace/ndispatch/$DS/scene_manifest.json
VR=workspace/overnight/vggt_${DS}_u100g
mkdir -p $VR logs
cp -f $MAN $VR/scene_manifest.json
SUM=workspace/overnight/${DS}_gated_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

pat "VGGT-gated $DS train start (epochs=10, pose_gate active)"
if ! $PY diagnostics/free_geometry/train_arms.py --run_root $VR --arms C2M_rawrel \
    --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components \
    > logs/${DS}_vggt_gated_train.log 2>&1; then
  pat "VGGT-gated $DS TRAIN FAILED (see logs/${DS}_vggt_gated_train.log)"; exit 1
fi
pat "VGGT-gated $DS train OK; eval start"
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
if $PY scripts/eval_vggt_cosw1.py --dataset $DS --scenes $SCENES \
    --run_root $VR --arm C2M_rawrel --step 100 --manifest $MAN \
    > logs/${DS}_vggt_gated_eval.log 2>&1; then
  pat "VGGT-gated $DS eval OK -> $VR/vggt_cosw1_eval.json"
else
  pat "VGGT-gated $DS EVAL FAILED (see logs/${DS}_vggt_gated_eval.log)"
fi
rm -rf $VR/eval32 $VR/ckpts 2>/dev/null
pat "==== VGGT-GATED $DS ALL DONE ===="
