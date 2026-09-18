#!/bin/bash
# Protocol v2 FULL lane B: VGGT on eth3d -> 7scenes -> scannetpp -> hiroom.
# A/B scene manifests (eval frames byte-identical to baseline manifests).
# After training: eval final step100 + eval selector-materialized ckpts.
# NEVER reruns baseline (SKIP_BASELINE=1).
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
V2="--v2_ab --v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_couple_fix --v2_rel_gate_deg 30 --loss_all_pos --v2_baselines_json workspace/protocol_v2/baselines.json"
STOPF=workspace/protocol_v2/STOP
ARM=C2M_RKDC1H

mkdir -p logs
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
  [ "${used:-99999}" -lt 60000 ] && break
  echo "[v3-vggt] waiting for GPU (used ${used}MiB) $(date '+%F %T')"; sleep 60
done
$PY $DG/train_arms.py --help 2>&1 | grep -q -- '--v2_rel_gate_deg' \
  || { echo "[v3-vggt] FATAL: rotation-v2 flags missing"; exit 1; }

view_subset () { case "$1" in eth3d|hiroom) echo allv;; *) echo 100v;; esac; }

run_ds () {
  local ds=$1
  local RR=workspace/protocol_v2/full_vggt_$ds
  local VS
  VS=$(view_subset "$ds")
  [ -f "$STOPF" ] && exit 1
  mkdir -p "$RR"
  cp "workspace/protocol_v2/ab_scene_manifests/$ds/scene_manifest.json" "$RR/scene_manifest.json"
  echo "[v3-vggt] $ds train START $(date '+%F %T')"
  # shellcheck disable=SC2086
  $PY $DG/train_arms.py --run_root "$RR" --arms $ARM --epochs 10 --seed 0 --no_eval32 \
      $V2 --swanlab --swanlab_suffix _v2full \
      >> "logs/v3_vggt_${ds}.log" 2>&1 \
    || { echo "[v3-vggt] $ds train FAILED $(date '+%F %T')"; return 1; }

  # pass 1: final-step eval
  SKIP_BASELINE=1 $PY $DG/eval_viewcounts.py --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms $ARM --view_subsets "$VS" >> "logs/v3_vggt_${ds}.log" 2>&1 || return 1
  $PY $DG/run_eval.py --run_root "$RR" --datas "$ds" \
      --experiments "${ARM}@${VS}" >> "logs/v3_vggt_${ds}.log" 2>&1 || return 1
  mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_V2final.json"
  rm -rf "$RR"/eval32/*/model_results

  # pass 2: selector-materialized ckpt eval (the applied fallback/selection);
  # selected ckpts live under arm name ${ARM}_SEL in ckpts_selected/
  $PY scripts/v2_materialize_selected.py --run_root "$RR" --arm $ARM --out_arm "${ARM}_SEL" \
      >> "logs/v3_vggt_${ds}.log" 2>&1 || return 1
  SKIP_BASELINE=1 $PY $DG/eval_viewcounts.py --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts_selected" --run_root "$RR" --step 100 \
      --arms "${ARM}_SEL" --view_subsets "$VS" \
      >> "logs/v3_vggt_${ds}.log" 2>&1 || return 1
  $PY $DG/run_eval.py --run_root "$RR" --datas "$ds" \
      --experiments "${ARM}_SEL@${VS}" >> "logs/v3_vggt_${ds}.log" 2>&1 || return 1
  mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_V2selected.json"
  rm -rf "$RR"/eval32/*/model_results

  $PY scripts/protocol_v2_analyze.py --model vggt --dataset "$ds" --run_dir "$RR" \
      >> "logs/v3_vggt_${ds}_analysis.log" 2>&1 || true
  echo "[v3-vggt] $ds DONE $(date '+%F %T')"
}

run_ds eth3d
run_ds 7scenes
run_ds scannetpp
run_ds hiroom
echo "[v3-vggt] ALL DONE $(date '+%F %T')"
