#!/bin/bash
# VGGT feat-weight ablation on dtu (chamfer dataset), no-mask, u100:
# w=0.0 (rel-only) and w=0.1 (weak feat). Everything else = campaign recipe.
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
MAN=workspace/ndispatch/dtu/scene_manifest.json
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
SUM=workspace/overnight/featw_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

W=$1
VR=workspace/overnight/vggt_dtu_nomask_w${W}_u100
mkdir -p $VR logs; cp -f $MAN $VR/scene_manifest.json
pat "VGGT feat_w=$W (nomask) train start"
if ! $PY diagnostics/free_geometry/train_arms.py --run_root $VR --arms C2M_rawrel \
    --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components \
    --no_pose_gate --no_mask --feat_weight $W \
    > logs/featw_w${W}_train.log 2>&1; then
  pat "feat_w=$W TRAIN FAILED"; exit 1
fi
pat "feat_w=$W eval start"
if $PY scripts/eval_vggt_cosw1.py --dataset dtu --scenes $SCENES \
    --run_root $VR --arm C2M_rawrel --step 100 --manifest $MAN \
    > logs/featw_w${W}_eval.log 2>&1; then
  pat "feat_w=$W eval OK -> $VR/vggt_cosw1_eval.json"
else
  pat "feat_w=$W EVAL FAILED"
fi
rm -rf $VR/eval32 $VR/ckpts 2>/dev/null
pat "==== FEATW w$W ALL DONE ===="
