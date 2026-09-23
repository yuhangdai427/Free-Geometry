#!/bin/bash
# OVERNIGHT: channel CWD (raw + LN) × 4 datasets × all scenes, with camrel.
# DA3 only; VGGT queued separately.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
ROOT=workspace/cwd_overnight
mkdir -p $ROOT
SUM=$ROOT/SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

run_variant() {
  local TAG="$1"; local MODE="$2"; local OUT="$ROOT/$TAG"
  for DS in eth3d 7scenes scannetpp hiroom; do
    MAN=artifacts/diagnostics/final_protocol/$DS/scene_manifest.json
    [ -f "$MAN" ] || { pat "$DS no manifest, skip"; continue; }
    SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
    for SC in $SCENES; do
      SL=$(echo "$SC" | tr '/' '_')
      pat "$TAG $DS/$SC start"
      if $PY scripts/train_pw0_accum.py --dataset $DS --scene $SC --vggt_sync \
          --half_mode $MODE --cwd_tau 0.5 --camrel \
          --updates 100 --accum 1 --manifest $MAN \
          --out $OUT/$DS/$SC > logs/cwdov_${TAG}_${DS}_${SL}.log 2>&1; then
        grep "cam_dec" logs/cwdov_${TAG}_${DS}_${SL}.log >> $SUM
      else
        pat "$TAG $DS/$SC FAILED"
      fi
      rm -rf $OUT/$DS/$SC/ckpts $OUT/$DS/$SC/recon
    done
  done
}

run_variant chcwd_raw  chcwd
run_variant chcwd_ln   chcwd_ln
pat "==== CWDOV ALL DONE ===="
