#!/bin/bash
# OVERNIGHT: raw-space patch + rel-pose recipe, 2 models x 4 datasets x 2 lengths
# (100/50 steps), manifest protocol as-is (teacher16/8 per manifest, spikes KEPT,
# clip 1.0), full eval per cell, per-cell disk cleanup, rc-retry once on OOM.
# dtu SKIPPED: no final_protocol manifest exists for it.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
ROOT=${OV_ROOT:-workspace/overnight}
mkdir -p $ROOT
SUM=$ROOT/SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }

run_retry() {  # $1 log, rest = command
  local log="$1"; shift
  "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] && grep -qiE "out of memory|OutOfMemory|CUDA error" "$log"; then
    pat "OOM retry: $log"
    sleep 30
    "$@" > "$log" 2>&1; rc=$?
  fi
  return $rc
}

for DS in ${DATASETS:-eth3d 7scenes scannetpp hiroom}; do
  MAN=artifacts/diagnostics/final_protocol/$DS/scene_manifest.json
  [ -f "$MAN" ] || { pat "MISSING manifest $DS — skip"; continue; }
  SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
  pat "==== DATASET $DS : $SCENES ===="

  for STEPS in ${STEPS_LIST:-100 50}; do
    EP=$((STEPS / 10))   # VGGT: epochs x 10 pairs = steps

    # ---------- DA3 ----------
    for SC in $SCENES; do
      SC_LOG=$(echo "$SC" | tr '/' '_')   # hiroom scene names contain slashes
      OUT=$ROOT/da3_${DS}_u${STEPS}/$SC
      pat "DA3 $DS/$SC u$STEPS start"
      if run_retry ${OV_LOGS:-logs}/ov_da3_${DS}_${SC_LOG}_u${STEPS}.log \
          $PY scripts/train_pw0_accum.py --dataset $DS --scene $SC --vggt_sync \
              --half_mode vggt --manifest $MAN --updates $STEPS --accum 1 \
              --out $OUT; then
        grep "cam_dec" ${OV_LOGS:-logs}/ov_da3_${DS}_${SC_LOG}_u${STEPS}.log >> $SUM
      else
        pat "DA3 $DS/$SC u$STEPS FAILED rc!=0 (see log)"
      fi
      # disk hygiene: keep grad CSVs only
      rm -rf $OUT/ckpts $OUT/recon 2>/dev/null
      # learning diagnosis: L_rot first10 vs last10
      if [ -f $OUT/grad_rel.csv ]; then
        $PY - "$OUT" >> $SUM 2>/dev/null <<'PYEOF'
import csv, sys, statistics as st
d = sys.argv[1]
rl = list(csv.DictReader(open(d + "/grad_rel.csv")))
if len(rl) >= 20:
    f = st.mean(float(r["l_rot"]) for r in rl[:10]); l = st.mean(float(r["l_rot"]) for r in rl[-10:])
    flag = "  <-- NOT LEARNING (rot)" if l >= f else ""
    print(f"[diag] {d}: L_rot {f:.3f}->{l:.3f}{flag}")
PYEOF
      fi
    done

    # ---------- VGGT (skippable for repair reruns) ----------
    [ "${SKIP_VGGT:-0}" = "1" ] && continue
    VR=$ROOT/vggt_${DS}_u${STEPS}
    mkdir -p $VR
    cp $MAN $VR/scene_manifest.json
    pat "VGGT $DS u$STEPS train start (epochs=$EP)"
    if run_retry ${OV_LOGS:-logs}/ov_vggt_${DS}_u${STEPS}.log \
        $PY $DG/train_arms.py --run_root $VR --arms C2M_rawrel \
            --epochs $EP --seed 0 --no_eval32 --loss_all_pos --grad_components; then
      pat "VGGT $DS u$STEPS train OK"
    else
      pat "VGGT $DS u$STEPS TRAIN FAILED"
    fi
    # TTA only — zero-shot baselines are frozen in protocol_v2/baselines.json
    SBF="--skip_baseline"
    pat "VGGT $DS u$STEPS eval start"
    if run_retry ${OV_LOGS:-logs}/ov_vggt_eval_${DS}_u${STEPS}.log \
        $PY scripts/eval_vggt_cosw1.py --dataset $DS --scenes $SCENES \
            --run_root $VR --arm C2M_rawrel --step $STEPS --manifest $MAN $SBF; then
      grep -E "AUC@3" ${OV_LOGS:-logs}/ov_vggt_eval_${DS}_u${STEPS}.log >> $SUM
    else
      pat "VGGT $DS u$STEPS EVAL FAILED"
    fi
    rm -rf $VR/eval32 $VR/ckpts 2>/dev/null   # keep summary json
  done
done
pat "==== ALL DONE ===="
