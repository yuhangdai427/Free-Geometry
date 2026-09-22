#!/bin/bash
# Round 2 (serial): camrel weight 0.5 (0.5/0.5/0.25) -> camrel + R/T per-term
# grad cap 1.0 -> camtok w=0.1 (restarted), all on eth3d's 11 manifest scenes.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
MAN=artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json
ROOT=workspace/cam_campaign
SUM=$ROOT/SUMMARY.log
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

run_cell() {
  local TAG="$1"; local OUT="$2"; shift 2
  for SC in $SCENES; do
    pat "R2 $TAG $SC start"
    if $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt \
        --updates 100 --accum 1 --manifest $MAN --out $OUT/$SC "$@" \
        > logs/cam_${TAG}_${SC}.log 2>&1; then
      grep "cam_dec" logs/cam_${TAG}_${SC}.log >> $SUM
    else
      pat "R2 $TAG $SC FAILED"
    fi
    rm -rf $OUT/$SC/ckpts $OUT/$SC/recon 2>/dev/null
  done
}

run_cell camrel05cap $ROOT/camrel05cap --camrel --camrel_rw 0.5 --camrel_tw 0.5 \
    --camrel_fw 0.25 --camrel_cap 1.0
run_cell camtok  $ROOT/camtok  --camtok --camtok_w 0.1
pat "==== ROUND2 ALL DONE ===="
