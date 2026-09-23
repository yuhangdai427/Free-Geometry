#!/bin/bash
# scannetpp TTA training (C2M_rawrel campaign recipe), ckpts RETAINED for the
# post-training probe. Usage: run_scannetpp_tta.sh da3|vggt
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
MAN=workspace/ndispatch/scannetpp/scene_manifest.json
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
SUM=workspace/overnight/scnpp_tta_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

da3() {
  for SC in $SCENES; do
    OUT=workspace/overnight/da3_scnpp_u100/$SC
    pat "DA3 $SC start"
    if $PY scripts/train_pw0_accum.py --dataset scannetpp --scene $SC --vggt_sync \
        --half_mode vggt --manifest $MAN --updates 100 --accum 1 --out $OUT \
        > logs/scnpp_da3_${SC}.log 2>&1; then
      pat "DA3 $SC OK (ckpt kept)"
    else
      pat "DA3 $SC FAILED"
    fi
    # NOTE: ckpts/ kept on purpose for the post-adaptation probe
  done
  pat "==== SCNPP DA3 TTA ALL DONE ===="
}

vggt() {
  VR=workspace/overnight/vggt_scnpp_u100
  mkdir -p $VR; cp -f $MAN $VR/scene_manifest.json
  pat "VGGT train start"
  if ! $PY diagnostics/free_geometry/train_arms.py --run_root $VR --arms C2M_rawrel \
      --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components --no_pose_gate \
      > logs/scnpp_vggt_train.log 2>&1; then
    pat "VGGT TRAIN FAILED"; return 1
  fi
  pat "==== SCNPP VGGT TTA ALL DONE (ckpts kept) ===="
}

case "${1:-}" in
  da3) da3 ;;
  vggt) vggt ;;
esac
