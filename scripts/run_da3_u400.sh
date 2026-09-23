#!/bin/bash
# DA3 200-update extension on dtu (chamfer trend still rising at u100).
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
MAN=workspace/ndispatch/dtu/scene_manifest.json
SUM=workspace/overnight/u400_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
for SC in $SCENES; do
  OUT=workspace/overnight/da3_dtu_u400/$SC
  pat "DA3-u400 $SC start"
  if $PY scripts/train_pw0_accum.py --dataset dtu --scene $SC --vggt_sync \
      --half_mode vggt --manifest $MAN --updates 400 --accum 1 --out $OUT \
      > logs/dtu_da3_u400_${SC}.log 2>&1; then
    grep "cam_dec" logs/dtu_da3_u400_${SC}.log >> $SUM
  else
    pat "DA3-u400 $SC FAILED"
  fi
  rm -rf $OUT/ckpts $OUT/recon 2>/dev/null
done
pat "==== DA3-U200 ALL DONE ===="
