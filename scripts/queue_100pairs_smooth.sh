#!/bin/bash
# DA3 scannetpp: 100 unique pairs × 100 steps, SmoothL1+2cos+camrel (vggt mode)
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
MAN=artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json
ROOT=workspace/spp100smooth
SUM=$ROOT/SUMMARY.log
mkdir -p $ROOT

SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
for SC in $SCENES; do
  echo "[$(date '+%F %T')] smooth100p $SC start" >> $SUM
  if $PY scripts/train_pw0_accum.py --dataset scannetpp --scene $SC \
      --vggt_sync --half_mode vggt --camrel \
      --n_train 100 --updates 100 --accum 1 --manifest $MAN \
      --out $ROOT/$SC > logs/spp100smooth_$SC.log 2>&1; then
    grep "cam_dec" logs/spp100smooth_$SC.log >> $SUM
  else
    echo "[$(date '+%F %T')] smooth100p $SC FAILED" >> $SUM
  fi
  rm -rf $ROOT/$SC/ckpts $ROOT/$SC/recon
done
echo "[$(date '+%F %T')] ==== SMOOTH100 ALL DONE ====" >> $SUM
