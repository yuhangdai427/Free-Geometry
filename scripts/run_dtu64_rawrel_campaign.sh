#!/bin/bash
# DTU-64 raw+rel TTA campaign (2026-09-22): DA3 + VGGT x 13 scenes (pose-only),
# recipe identical to run_dtu_rawrel_campaign.sh. eval3d raises NotImplemented
# -> recon skipped, pose AUC is the metric.
# Usage: run_dtu64_rawrel_campaign.sh da3lane|vggtlane
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
MAN=workspace/ndispatch/dtu64/scene_manifest.json
ROOT=workspace/overnight
SCENES="scan105 scan114 scan118 scan122 scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97"
SUM=$ROOT/dtu64_rawrel_SUMMARY.log
mkdir -p $ROOT logs
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

da3lane() {
  for SC in $SCENES; do
    OUT=$ROOT/da3_dtu64_u100/$SC
    pat "DA3 $SC start"
    if $PY scripts/train_pw0_accum.py --dataset dtu64 --scene $SC --vggt_sync \
        --half_mode vggt --manifest $MAN --updates 100 --accum 1 --out $OUT \
        > logs/dtu64_da3_${SC}.log 2>&1; then
      grep "cam_dec" logs/dtu64_da3_${SC}.log >> $SUM
    else
      pat "DA3 $SC FAILED rc!=0 (see logs/dtu64_da3_${SC}.log)"
    fi
    rm -rf $OUT/ckpts $OUT/recon 2>/dev/null
  done
  pat "DA3 LANE DONE"
}

vggtlane() {
  VR=$ROOT/vggt_dtu64_u100
  mkdir -p $VR
  cp -f $MAN $VR/scene_manifest.json
  pat "VGGT train start (epochs=10 -> 100 steps)"
  if ! $PY diagnostics/free_geometry/train_arms.py --run_root $VR --arms C2M_rawrel \
      --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components \
      > logs/dtu64_vggt_train.log 2>&1; then
    pat "VGGT TRAIN FAILED"; return 1
  fi
  pat "VGGT train OK; eval start"
  if $PY scripts/eval_vggt_cosw1.py --dataset dtu64 --scenes $SCENES \
      --run_root $VR --arm C2M_rawrel --step 100 --manifest $MAN \
      > logs/dtu64_vggt_eval.log 2>&1; then
    pat "VGGT eval OK -> $VR/vggt_cosw1_eval.json"
  else
    pat "VGGT EVAL FAILED (see logs/dtu64_vggt_eval.log)"
  fi
  rm -rf $VR/eval32 $VR/ckpts 2>/dev/null
  pat "VGGT LANE DONE"
}

case "${1:-}" in
  da3lane) da3lane ;;
  vggtlane) vggtlane ;;
  *) echo "usage: $0 da3lane|vggtlane"; exit 1 ;;
esac
pat "==== ${1} ALL DONE ===="
