#!/bin/bash
# DA3 zero-update (TTA-free) baseline evals: dtu (22) + dtu64 (13), same
# evaluator as the TTA campaign so gains are apples-to-apples.
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
SUM=workspace/overnight/baseline_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

run_ds() {
  local DS=$1; shift
  local MAN=workspace/ndispatch/$DS/scene_manifest.json
  for SC in "$@"; do
    local OUT=workspace/overnight/da3_${DS}_u0/$SC
    pat "BASE $DS/$SC start"
    if $PY scripts/train_pw0_accum.py --dataset $DS --scene $SC --vggt_sync \
        --half_mode vggt --manifest $MAN --updates 0 --accum 1 --out $OUT \
        > logs/${DS}_da3_u0_${SC}.log 2>&1; then
      grep "cam_dec" logs/${DS}_da3_u0_${SC}.log >> $SUM
    else
      pat "BASE $DS/$SC FAILED (see logs/${DS}_da3_u0_${SC}.log)"
    fi
    rm -rf $OUT/ckpts $OUT/recon 2>/dev/null
  done
}

run_ds dtu scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 \
          scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118
run_ds dtu64 scan105 scan114 scan118 scan122 scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97
pat "==== BASELINE ALL DONE ===="
