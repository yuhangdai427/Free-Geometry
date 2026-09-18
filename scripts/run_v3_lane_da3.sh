#!/bin/bash
# Protocol v2 FULL (layers 1+2+3+4+5): A/B contexts + reliability weights +
# robust R/T + grad cap + couple fix + probe/ckpt/selector + rel gate + ramp.
# Lane A: DA3 on eth3d -> 7scenes -> scannetpp -> hiroom -> dtu -> dtu64.
# NEVER reruns baseline. Requires: ab_manifests built, wiring flags present.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
MAN=workspace/protocol_v2/ab_manifests
V2="--v2_ab_manifest $MAN --v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_couple_fix --v2_rel_gate_deg 0 --v2_apply_selection --v2_baselines_json workspace/protocol_v2/baselines.json"
STOPF=workspace/protocol_v2/STOP

mkdir -p logs
# preflight: GPU has capacity, flags exist, manifests exist
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
  [ "${used:-99999}" -lt 60000 ] && break
  echo "[v3-da3] waiting for GPU (used ${used}MiB) $(date '+%F %T')"; sleep 60
done
$PY scripts/train_da3_protocol.py --help 2>&1 | grep -q -- '--v2_rel_gate_deg' \
  || { echo "[v3-da3] FATAL: rotation-v2 flags missing"; exit 1; }
[ -d "$MAN/eth3d" ] || { echo "[v3-da3] FATAL: no ab manifests"; exit 1; }

scenes_of () {
  $PY - "$1" <<'EOF'
import glob, json, sys
for p in sorted(glob.glob(f"workspace/protocol_v2/ab_manifests/{sys.argv[1]}/*.json")):
    try:
        print(json.load(open(p))["scene"])
    except Exception:
        pass
EOF
}

run_ds () {
  local ds=$1
  local out=workspace/protocol_v2/full_da3_$ds
  [ -f "$STOPF" ] && { echo "[v3-da3] STOP"; exit 1; }
  # skip scenes whose final ckpt already exists (restart-safe)
  local scenes=""
  for sc in $(scenes_of "$ds" | tr '\n' ' '); do
    if [ -f "$out/ckpts/$sc/v2/step100_lora.pt" ]; then
      echo "[v3-da3] skip $sc (done)"
    else
      scenes="$scenes $sc"
    fi
  done
  [ -z "${scenes# }" ] && { echo "[v3-da3] $ds already complete"; return 0; }
  echo "[v3-da3] $ds START $(date '+%F %T') scenes:${scenes}"
  # shellcheck disable=SC2086
  $PY scripts/train_da3_protocol.py --dataset "$ds" --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 --n_train 10 \
      --mask_ratio 0.5 --loss_all_pos $V2 \
      --swanlab --swanlab_suffix _v2full \
      >> "logs/v3_da3_${ds}.log" 2>&1 \
    && echo "[v3-da3] $ds DONE $(date '+%F %T')" \
    || echo "[v3-da3] $ds FAILED $(date '+%F %T')"
  $PY scripts/protocol_v2_analyze.py --model da3 --dataset "$ds" --run_dir "$out" \
      >> "logs/v3_da3_${ds}_analysis.log" 2>&1 || true
}

run_ds eth3d
run_ds 7scenes
run_ds scannetpp
run_ds hiroom
run_ds dtu
run_ds dtu64
echo "[v3-da3] ALL DONE $(date '+%F %T')"
