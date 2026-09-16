#!/bin/bash
# Remaining-battery mega chain: strictly sequential, one GPU job at a time.
# Each unit logs "UNIT_OK <name>" or "UNIT_FAIL <name>" to remaining_chain.log.
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
DG=diagnostics/free_geometry
LOG=artifacts/diagnostics/remaining_chain.log
: > "$LOG"

run_unit () {  # $1=name $2=ROOT $3=variant $4=ds $5=views $6=ARM
  local NAME=$1 ROOTV=$2 V=$3 DS=$4 VS=$5 AR=$6
  echo "[$(date '+%F %T')] UNIT_START $NAME" >> "$LOG"
  if ROOT=$ROOTV ARM=$AR bash "$DG/run_variant.sh" "$V" "$DS" "$VS"; then
    echo "[$(date '+%F %T')] UNIT_OK $NAME" >> "$LOG"
  else
    echo "[$(date '+%F %T')] UNIT_FAIL $NAME" >> "$LOG"
  fi
}

run_unit s248b artifacts/diagnostics/final_protocol_s8 scannetpp_s8t24 scannetpp 100v B5_maskdistill

for ds in scannetpp 7scenes hiroom eth3d; do
  V=100v; [ "$ds" = "hiroom" ] || [ "$ds" = "eth3d" ] && V=allv
  run_unit cta_$ds artifacts/diagnostics/final_protocol_phase4 $ds $ds $V CTM_ANC
done

for A in C2M_TRIP C2M_TRIF C2M_TRIF2 C2M_TRIF3; do
  for ds in scannetpp 7scenes hiroom eth3d; do
    V=100v; [ "$ds" = "hiroom" ] || [ "$ds" = "eth3d" ] && V=allv
    run_unit tri_${A}_$ds artifacts/diagnostics/final_protocol_phase4 $ds $ds $V $A
  done
done

for A in C2M_SCL C2M_CYC C2M_CONFP; do
  for ds in scannetpp 7scenes hiroom eth3d; do
    V=100v; [ "$ds" = "hiroom" ] || [ "$ds" = "eth3d" ] && V=allv
    run_unit pb_${A}_$ds artifacts/diagnostics/final_protocol_phase4 $ds $ds $V $A
  done
done

for ds in scannetpp 7scenes hiroom eth3d; do
  V=100v; [ "$ds" = "hiroom" ] || [ "$ds" = "eth3d" ] && V=allv
  run_unit gate_$ds artifacts/diagnostics/final_protocol_phase4 $ds $ds $V C2M_GATE
done
for ds in scannetpp 7scenes; do
  run_unit mc_$ds artifacts/diagnostics/final_protocol_phase4 $ds $ds 100v C2M_MC
done

# single-scene checks on the wounded scene (1ada7a0617): train + eval per arm
RR=artifacts/diagnostics/final_protocol/scannetpp
for A in C2M_SCL C2M_CYC C2M_CONFP C2M_GATE C2M_MC; do
  echo "[$(date '+%F %T')] UNIT_START single_$A" >> "$LOG"
  if python "$DG/train_arms.py" --run_root "$RR" --arms "$A" --scenes 1ada7a0617 \
      --epochs 10 --seed 0 --no_eval32 >> "$RR/stream_single_$A.log" 2>&1 \
     && python "$DG/eval_one_scene.py" --run_root "$RR" --scene 1ada7a0617 \
      --arm "$A" --views 100v >> "$RR/stream_single_$A.log" 2>&1; then
    echo "[$(date '+%F %T')] UNIT_OK single_$A" >> "$LOG"
  else
    echo "[$(date '+%F %T')] UNIT_FAIL single_$A" >> "$LOG"
  fi
done

echo "[$(date '+%F %T')] CHAIN_ALL_DONE" >> "$LOG"
