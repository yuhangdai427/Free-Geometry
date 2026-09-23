#!/bin/bash
# NO-MASK variants: channel CWD (raw + LN) + camrel, CLEAN student input.
# Queued after the main overnight campaign finishes.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
ROOT=workspace/cwd_overnight
SUM=$ROOT/SUMMARY.log

# wait for main campaign
for i in $(seq 1 4000); do
  if ! pgrep -f "train_pw0_accum" >/dev/null 2>&1 \
     && grep -q "CWDOV ALL DONE" $SUM 2>/dev/null; then
    break
  fi
  sleep 30
done

run_variant() {
  local TAG="$1"; local MODE="$2"
  for DS in eth3d 7scenes scannetpp hiroom; do
    MAN=artifacts/diagnostics/final_protocol/$DS/scene_manifest.json
    [ -f "$MAN" ] || continue
    SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
    for SC in $SCENES; do
      SL=$(echo "$SC" | tr '/' '_')
      echo "[$(date '+%F %T')] nomask_$TAG $DS/$SC start" >> $SUM
      if $PY scripts/train_pw0_accum.py --dataset $DS --scene $SC --vggt_sync \
          --half_mode $MODE --cwd_tau 0.5 --camrel --no_mask \
          --updates 100 --accum 1 --manifest $MAN \
          --out $ROOT/nomask_$TAG/$DS/$SC > logs/cwdov_nomask_${TAG}_${DS}_${SL}.log 2>&1; then
        grep "cam_dec" logs/cwdov_nomask_${TAG}_${DS}_${SL}.log >> $SUM
      else
        echo "[$(date '+%F %T')] nomask_$TAG $DS/$SC FAILED" >> $SUM
      fi
      rm -rf $ROOT/nomask_$TAG/$DS/$SC/ckpts $ROOT/nomask_$TAG/$DS/$SC/recon
    done
  done
}

run_variant chcwd_raw chcwd
run_variant chcwd_ln  chcwd_ln
echo "[$(date '+%F %T')] ==== NOMASK ALL DONE ====" >> $SUM
