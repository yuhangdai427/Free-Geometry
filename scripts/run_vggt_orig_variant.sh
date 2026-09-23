#!/bin/bash
# VGGT original (ungated) rawrel variant. Usage: run_vggt_orig_variant.sh <ds> <steps> <tag>
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
DS=$1; STEPS=$2; TAG=$3
EP=$((STEPS / 10))
MAN=workspace/ndispatch/$DS/scene_manifest.json
VR=workspace/overnight/vggt_${DS}_${TAG}
mkdir -p $VR logs
cp -f $MAN $VR/scene_manifest.json
SUM=workspace/overnight/${DS}_${TAG}_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pat "VGGT-$TAG $DS train start (epochs=$EP, no_pose_gate)"
if ! $PY diagnostics/free_geometry/train_arms.py --run_root $VR --arms C2M_rawrel \
    --epochs $EP --seed 0 --no_eval32 --loss_all_pos --grad_components --no_pose_gate \
    > logs/${DS}_vggt_${TAG}_train.log 2>&1; then
  pat "TRAIN FAILED"; exit 1
fi
pat "train OK; eval start"
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
if $PY scripts/eval_vggt_cosw1.py --dataset $DS --scenes $SCENES \
    --run_root $VR --arm C2M_rawrel --step $STEPS --manifest $MAN \
    > logs/${DS}_vggt_${TAG}_eval.log 2>&1; then
  pat "eval OK -> $VR/vggt_cosw1_eval.json"
else
  pat "EVAL FAILED"
fi
rm -rf $VR/eval32 $VR/ckpts 2>/dev/null
pat "==== VGGT-$TAG $DS ALL DONE ===="
