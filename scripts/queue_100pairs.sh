#!/bin/bash
# DA3 scannetpp: 100 unique training pairs × 100 steps (1 visit per pair),
# channel CWD raw + camrel, with mask. All 20 scenes.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
MAN=artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json
ROOT=workspace/scannetpp_100pairs
SUM=$ROOT/SUMMARY.log
mkdir -p $ROOT

SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
for SC in $SCENES; do
  echo "[$(date '+%F %T')] 100p $SC start" >> $SUM
  if $PY scripts/train_pw0_accum.py --dataset scannetpp --scene $SC \
      --vggt_sync --half_mode chcwd --cwd_tau 0.5 --camrel \
      --n_train 100 --updates 100 --accum 1 --manifest $MAN \
      --out $ROOT/$SC > logs/spp100_$SC.log 2>&1; then
    grep "cam_dec" logs/spp100_$SC.log >> $SUM
  else
    echo "[$(date '+%F %T')] 100p $SC FAILED" >> $SUM
  fi
  rm -rf $ROOT/$SC/ckpts $ROOT/$SC/recon
done
echo "[$(date '+%F %T')] ==== 100P ALL DONE ====" >> $SUM
