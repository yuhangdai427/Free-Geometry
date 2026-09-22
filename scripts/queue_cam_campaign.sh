#!/bin/bash
# camrel v1 (1/1/0.5) + camtok (w=0.1) on ALL eth3d manifest scenes (11),
# u100k1, manifest protocol, cam_dec eval per run. SERIAL; disk-clean per run.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
MAN=artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json
ROOT=workspace/cam_campaign
mkdir -p $ROOT
SUM=$ROOT/SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

run_cell() {  # $1=tag $2=outdir $3...=extra args
  local TAG="$1"; shift
  local OUT="$1"; shift
  for SC in $($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))"); do
    pat "$TAG $SC start"
    if $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt \
        --updates 100 --accum 1 --manifest $MAN --out $OUT/$SC "$@" \
        > logs/cam_${TAG}_${SC}.log 2>&1; then
      grep "cam_dec" logs/cam_${TAG}_${SC}.log >> $SUM
    else
      if grep -qiE "out of memory|OutOfMemory" logs/cam_${TAG}_${SC}.log; then
        pat "$TAG $SC OOM retry"
        sleep 30
        $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt \
            --updates 100 --accum 1 --manifest $MAN --out $OUT/$SC "$@" \
            > logs/cam_${TAG}_${SC}.log 2>&1 \
            && grep "cam_dec" logs/cam_${TAG}_${SC}.log >> $SUM \
            || pat "$TAG $SC FAILED-after-retry"
      else
        pat "$TAG $SC FAILED"
      fi
    fi
    rm -rf $OUT/$SC/ckpts $OUT/$SC/recon 2>/dev/null
  done
}

run_cell camrel $ROOT/camrel --camrel
run_cell camtok  $ROOT/camtok  --camtok --camtok_w 0.1
pat "==== ALL DONE ===="
