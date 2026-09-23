#!/bin/bash
# No-mask ablation on dtu (chamfer dataset): DA3 lane (u300 -> u100) and
# VGGT lane (u100 -> u50), --no_mask, everything else identical to the
# campaign recipe. Runs inside tmux session fgnomask (laneA/laneB windows).
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
MAN=workspace/ndispatch/dtu/scene_manifest.json
SCENES="scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118"
SUM=workspace/overnight/nomask_SUMMARY.log
mkdir -p logs
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

da3_lane() {
  for U in 300 100; do
    for SC in $SCENES; do
      OUT=workspace/overnight/da3_dtu_nomask_u${U}/$SC
      pat "DA3-nomask u$U $SC start"
      if $PY scripts/train_pw0_accum.py --dataset dtu --scene $SC --vggt_sync \
          --half_mode vggt --manifest $MAN --updates $U --accum 1 --no_mask \
          --out $OUT > logs/nomask_da3_u${U}_${SC}.log 2>&1; then
        grep "cam_dec" logs/nomask_da3_u${U}_${SC}.log >> $SUM
      else
        pat "DA3-nomask u$U $SC FAILED (see logs/nomask_da3_u${U}_${SC}.log)"
      fi
      rm -rf $OUT/ckpts $OUT/recon 2>/dev/null
    done
    pat "DA3-nomask u$U LANE SEGMENT DONE"
  done
  pat "==== DA3-NOMASK ALL DONE ===="
}

vggt_lane() {
  for CFG in "100 10" "50 5"; do
    set -- $CFG; U=$1; EP=$2
    VR=workspace/overnight/vggt_dtu_nomask_u${U}
    mkdir -p $VR; cp -f $MAN $VR/scene_manifest.json
    pat "VGGT-nomask u$U train start (epochs=$EP)"
    if ! $PY diagnostics/free_geometry/train_arms.py --run_root $VR --arms C2M_rawrel \
        --epochs $EP --seed 0 --no_eval32 --loss_all_pos --grad_components \
        --no_pose_gate --no_mask \
        > logs/nomask_vggt_u${U}_train.log 2>&1; then
      pat "VGGT-nomask u$U TRAIN FAILED"; continue
    fi
    pat "VGGT-nomask u$U eval start"
    if $PY scripts/eval_vggt_cosw1.py --dataset dtu --scenes $SCENES \
        --run_root $VR --arm C2M_rawrel --step $U --manifest $MAN \
        > logs/nomask_vggt_u${U}_eval.log 2>&1; then
      pat "VGGT-nomask u$U eval OK -> $VR/vggt_cosw1_eval.json"
    else
      pat "VGGT-nomask u$U EVAL FAILED"
    fi
    rm -rf $VR/eval32 $VR/ckpts 2>/dev/null
  done
  pat "==== VGGT-NOMASK ALL DONE ===="
}

case "${1:-}" in
  da3) da3_lane ;;
  vggt) vggt_lane ;;
  *) echo "usage: $0 da3|vggt"; exit 1 ;;
esac
