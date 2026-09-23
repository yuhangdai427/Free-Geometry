#!/bin/bash
# VGGT channel CWD (raw + LN) × 4 datasets, queued after DA3 chcwd_ln
# finishes and BEFORE the no-mask experiments.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
ROOT=workspace/cwd_overnight
SUM=$ROOT/SUMMARY.log

# wait for DA3 chcwd_ln to finish (CWDOV ALL DONE marker)
for i in $(seq 1 3000); do
  if ! pgrep -f "train_pw0_accum" >/dev/null 2>&1 \
     && grep -q "CWDOV ALL DONE" $SUM 2>/dev/null; then
    break
  fi
  sleep 30
done

for ARM in C2M_chcwd C2M_chcwd_ln; do
  TAG=$(echo $ARM | sed 's/C2M_//')
  VR=$ROOT/vggt_$TAG
  mkdir -p $VR
  for DS in eth3d 7scenes scannetpp hiroom; do
    MAN=artifacts/diagnostics/final_protocol/$DS/scene_manifest.json
    [ -f "$MAN" ] || continue
    cp $MAN $VR/scene_manifest.json
    echo "[$(date '+%F %T')] VGGT $TAG $DS start" >> $SUM
    if $PY $DG/train_arms.py --run_root $VR --arms $ARM \
        --scenes "" --epochs 10 --seed 0 --no_eval32 --loss_all_pos \
        > logs/cwdov_vggt_${TAG}_${DS}.log 2>&1; then
      echo "[$(date '+%F %T')] VGGT $TAG $DS train OK" >> $SUM
    else
      echo "[$(date '+%F %T')] VGGT $TAG $DS TRAIN FAILED" >> $SUM
    fi
    # eval
    SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
    $PY scripts/eval_vggt_cosw1.py --dataset $DS --scenes "$SCENES" \
        --run_root $VR --arm $ARM --step 100 --manifest $MAN --skip_baseline \
        > logs/cwdov_vggt_eval_${TAG}_${DS}.log 2>&1 \
        && grep "TTA:" logs/cwdov_vggt_eval_${TAG}_${DS}.log >> $SUM
    rm -rf $VR/eval32 $VR/ckpts
  done
done
echo "[$(date '+%F %T')] ==== VGGT CHCWD ALL DONE ====" >> $SUM
