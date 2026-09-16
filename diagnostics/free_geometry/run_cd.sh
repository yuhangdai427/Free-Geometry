#!/bin/bash
# Rerun of failed phases C and D (fixed scene-list expansion).
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
D=diagnostics/free_geometry
A=artifacts/diagnostics
LOG=$A/overnight_cd.log

say()  { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
guard() {
  local name="$1"; shift
  local free_gb=$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt 30 ]; then say "ABORT $name: only ${free_gb}G disk left"; return 1; fi
  say "START $name"
  if "$@" >> $LOG 2>&1; then say "OK $name"; else say "FAILED $name (see log)"; fi
}

SCENES=$(python3 $D/_six_scenes.py)
cp $A/bakeoff_v2_transductive/scene_manifest.json $A/bakeoff_v7_mech/scene_manifest.json 2>/dev/null || (mkdir -p $A/bakeoff_v7_mech && cp $A/bakeoff_v2_transductive/scene_manifest.json $A/bakeoff_v7_mech/)
mkdir -p $A/bakeoff_v8_forms && cp $A/bakeoff_v2_transductive/scene_manifest.json $A/bakeoff_v8_forms/
say "C/D scenes: $SCENES"

guard "C3 full-finetune" python3 $D/train_arms.py --run_root $A/bakeoff_v7_mech \
  --scenes $SCENES --arms B5_conf --full_ft --no_ckpt
guard "C5 camtok unfrozen" python3 $D/train_arms.py --run_root $A/bakeoff_v7_mech \
  --scenes $SCENES --arms C2_b5_rel --train_camera_token --no_ckpt
guard "C4 selfmix" python3 $D/train_arms.py --run_root $A/bakeoff_v7_mech \
  --scenes $SCENES --arms C4_selfmix --no_ckpt
guard "C eval" python3 $D/run_eval.py --run_root $A/bakeoff_v7_mech \
  --manifest $A/bakeoff_v2_transductive/scene_manifest.json

guard "D new forms" python3 $D/train_arms.py --run_root $A/bakeoff_v8_forms \
  --scenes $SCENES --arms D1_relfeat D2_layermean D3_spatialpool D4_depthgrad --no_ckpt
guard "D eval" python3 $D/run_eval.py --run_root $A/bakeoff_v8_forms \
  --manifest $A/bakeoff_v2_transductive/scene_manifest.json
guard "D depth metrics" python3 $D/depth_metrics.py --run_root $A/bakeoff_v8_forms \
  --manifest $A/bakeoff_v2_transductive/scene_manifest.json

say "=== C/D rerun finished $(date) ==="
