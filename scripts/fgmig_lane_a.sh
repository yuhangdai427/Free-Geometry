#!/bin/bash
# Lane A: VGGT-Omega (starts immediately, runs alongside the lossall chain)
# then DVLT (waits for its env marker AND omega completion). Serial within lane.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
RUNS=workspace/fgmig/runs

run () {  # model ds arm [extra args]
  local model=$1 ds=$2 arm=$3; shift 3
  local out=$RUNS/${model}_${ds}_${arm}
  if [ -f "$out/metrics_attempted" ]; then echo "[laneA] skip $out (attempted)"; return; fi
  touch "$out/metrics_attempted" 2>/dev/null || mkdir -p "$out" && touch "$out/metrics_attempted"
  echo "[laneA] $model $ds $arm START $(date '+%F %T')"
  $PY scripts/train_fg_protocol.py --model $model --dataset $ds --arm $arm \
      --output_root "$out" "$@" >> logs/fgmig_${model}_${ds}_${arm}.log 2>&1 \
      || { echo "[laneA] $model $ds $arm TRAIN-FAILED"; return 1; }
  echo "[laneA] $model $ds $arm EXPORT-DONE $(date '+%F %T')"
}

# ---- baselines first (fast, eval-only) ----
for ds in 7scenes eth3d hiroom scannetpp; do run omega $ds a0; done
# ---- TTA arms ----
for ds in 7scenes eth3d hiroom scannetpp; do run omega $ds rkdc_allpos; done
run omega 7scenes m_allpos

# ---- DVLT after env ready ----
echo "[laneA] waiting for DVLT env (logs/dvlt_pip.log DVLT-PIP-OK) $(date '+%F %T')"
for i in $(seq 1 360); do grep -q "DVLT-PIP-OK" logs/dvlt_pip.log 2>/dev/null && break; sleep 30; done
if ! grep -q "DVLT-PIP-OK" logs/dvlt_pip.log 2>/dev/null; then
  echo "[laneA] DVLT env never became ready — SKIPPING dvlt"; exit 0
fi
PYD=/root/autodl-tmp/conda/envs/dvlt/bin/python
export PYTHONPATH=/root/autodl-tmp/dvlt/src
run_dvlt () {
  local ds=$1 arm=$2
  local out=$RUNS/dvlt_${ds}_${arm}
  [ -f "$out/metrics_attempted" ] && { echo "[laneA] skip $out"; return; }
  mkdir -p "$out"; touch "$out/metrics_attempted"
  echo "[laneA] dvlt $ds $arm START $(date '+%F %T')"
  $PYD scripts/train_fg_protocol.py --model dvlt --dataset $ds --arm $arm \
      --output_root "$out" >> logs/fgmig_dvlt_${ds}_${arm}.log 2>&1 \
      || { echo "[laneA] dvlt $ds $arm TRAIN-FAILED"; return 1; }
  echo "[laneA] dvlt $ds $arm EXPORT-DONE $(date '+%F %T')"
}
for ds in 7scenes eth3d hiroom scannetpp; do run_dvlt $ds a0; done
for ds in 7scenes eth3d hiroom scannetpp; do run_dvlt $ds rkdc_allpos; done
echo "[laneA] ALL DONE $(date '+%F %T')"
