#!/bin/bash
# DA3 50-update variant (step ablation): dtu + dtu64, same recipe otherwise.
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
SUM=workspace/overnight/u50_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

for DS in dtu dtu64; do
  MAN=workspace/ndispatch/$DS/scene_manifest.json
  SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
  for SC in $SCENES; do
    OUT=workspace/overnight/da3_${DS}_u50/$SC
    pat "DA3-u50 $DS/$SC start"
    if $PY scripts/train_pw0_accum.py --dataset $DS --scene $SC --vggt_sync \
        --half_mode vggt --manifest $MAN --updates 50 --accum 1 --out $OUT \
        > logs/${DS}_da3_u50_${SC}.log 2>&1; then
      grep "cam_dec" logs/${DS}_da3_u50_${SC}.log >> $SUM
    else
      pat "DA3-u50 $DS/$SC FAILED (see logs/${DS}_da3_u50_${SC}.log)"
    fi
    rm -rf $OUT/ckpts $OUT/recon 2>/dev/null
  done
done
pat "==== DA3-U50 ALL DONE ===="
