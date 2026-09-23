#!/bin/bash
# VGGT channel CWD (raw + LN) × 4 datasets + rel-pose, all scenes.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
ROOT=workspace/vggt_chcwd
SUM=$ROOT/SUMMARY.log
mkdir -p $ROOT

for ARM in C2M_chcwd C2M_chcwd_ln; do
  TAG=$(echo $ARM | sed 's/C2M_//')
  VR=$ROOT/$TAG
  mkdir -p $VR
  for DS in eth3d 7scenes scannetpp hiroom; do
    MAN=artifacts/diagnostics/final_protocol/$DS/scene_manifest.json
    [ -f "$MAN" ] || continue
    cp $MAN $VR/scene_manifest.json
    SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
    echo "[$(date '+%F %T')] VGGT $TAG $DS train start ($SCENES)" >> $SUM
    if $PY $DG/train_arms.py --run_root $VR --arms $ARM \
        --scenes $SCENES --epochs 10 --seed 0 --no_eval32 --loss_all_pos \
        > logs/vggt_chcwd_${TAG}_${DS}.log 2>&1; then
      echo "[$(date '+%F %T')] VGGT $TAG $DS train OK" >> $SUM
    else
      echo "[$(date '+%F %T')] VGGT $TAG $DS TRAIN FAILED" >> $SUM
      tail -5 logs/vggt_chcwd_${TAG}_${DS}.log >> $SUM
      continue
    fi
    # eval (TTA only)
    echo "[$(date '+%F %T')] VGGT $TAG $DS eval start" >> $SUM
    $PY scripts/eval_vggt_cosw1.py --dataset $DS --scenes "$SCENES" \
        --run_root $VR --arm $ARM --step 100 --manifest $MAN --skip_baseline \
        > logs/vggt_chcwd_eval_${TAG}_${DS}.log 2>&1 \
        && grep "TTA:" logs/vggt_chcwd_eval_${TAG}_${DS}.log >> $SUM \
        || echo "[$(date '+%F %T')] VGGT $TAG $DS EVAL FAILED" >> $SUM
    rm -rf $VR/eval32 $VR/ckpts
  done
done
echo "[$(date '+%F %T')] ==== VGGT CHCWD ALL DONE ====" >> $SUM
