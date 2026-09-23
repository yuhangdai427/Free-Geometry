#!/bin/bash
# DTU raw+rel TTA campaign (2026-09-22): DA3 + VGGT x 22 scenes, 100 updates,
# recipe identical to scripts/overnight_rawrel.sh (C2M_rawrel / --half_mode vggt,
# manifest pairs, spikes kept, campaign clip), full-scene eval per cell
# (pose AUC + DTU chamfer acc/comp/overall), per-cell disk cleanup.
# Usage: run_dtu_rawrel_campaign.sh da3lane|vggtlane
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
MAN=workspace/ndispatch/dtu/scene_manifest.json
ROOT=workspace/overnight
SCENES="scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118"
SUM=$ROOT/dtu_rawrel_SUMMARY.log
mkdir -p $ROOT logs
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

da3lane() {
  for SC in $SCENES; do
    OUT=$ROOT/da3_dtu_u100/$SC
    pat "DA3 $SC start"
    if $PY scripts/train_pw0_accum.py --dataset dtu --scene $SC --vggt_sync \
        --half_mode vggt --manifest $MAN --updates 100 --accum 1 --out $OUT \
        > logs/dtu_da3_${SC}.log 2>&1; then
      grep "cam_dec" logs/dtu_da3_${SC}.log >> $SUM
    else
      pat "DA3 $SC FAILED rc!=0 (see logs/dtu_da3_${SC}.log)"
    fi
    rm -rf $OUT/ckpts $OUT/recon 2>/dev/null   # keep eval.json + grad CSVs
  done
  pat "DA3 LANE DONE"
}

vggtlane() {
  VR=$ROOT/vggt_dtu_u100
  mkdir -p $VR
  cp -f $MAN $VR/scene_manifest.json
  pat "VGGT train start (epochs=10 -> 100 steps)"
  if ! $PY diagnostics/free_geometry/train_arms.py --run_root $VR --arms C2M_rawrel \
      --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components \
      > logs/dtu_vggt_train.log 2>&1; then
    pat "VGGT TRAIN FAILED (see logs/dtu_vggt_train.log)"; return 1
  fi
  pat "VGGT train OK; eval start"
  if $PY scripts/eval_vggt_cosw1.py --dataset dtu --scenes $SCENES \
      --run_root $VR --arm C2M_rawrel --step 100 --manifest $MAN \
      > logs/dtu_vggt_eval.log 2>&1; then
    pat "VGGT eval OK -> $VR/vggt_cosw1_eval.json"
  else
    pat "VGGT EVAL FAILED (see logs/dtu_vggt_eval.log)"
  fi
  rm -rf $VR/eval32 $VR/ckpts 2>/dev/null   # keep vggt_cosw1_eval.json
  pat "VGGT LANE DONE"
}

case "${1:-}" in
  da3lane) da3lane ;;
  vggtlane) vggtlane ;;
  *) echo "usage: $0 da3lane|vggtlane"; exit 1 ;;
esac
pat "==== ${1} ALL DONE ===="
