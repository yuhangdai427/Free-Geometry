#!/bin/bash
# DA3 lane B (split): run a SUBSET of datasets in parallel with the main lane.
# Usage: DATASETS="hiroom dtu dtu64" bash scripts/run_v3_lane_da3_b.sh
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
MAN=workspace/protocol_v2/ab_manifests
V2="--v2_ab_manifest $MAN --v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_qfeat_off --v2_rel_gate_deg 0 --v2_apply_selection --v2_baselines_json workspace/protocol_v2/baselines.json"
STOPF=workspace/protocol_v2/STOP
DATASETS="${DATASETS:-hiroom}"

mkdir -p logs
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
  [ "${used:-99999}" -lt 60000 ] && break
  echo "[v3-da3-b] waiting for GPU (${used}MiB) $(date '+%F %T')"; sleep 60
done

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
  [ -f "$STOPF" ] && { echo "[v3-da3-b] STOP"; exit 1; }
  local scenes=""
  for sc in $(scenes_of "$ds" | tr '\n' ' '); do
    if [ -f "$out/ckpts/$sc/v2/step100_lora.pt" ]; then
      echo "[v3-da3-b] skip $sc (done)"
    else
      scenes="$scenes $sc"
    fi
  done
  [ -z "${scenes# }" ] && { echo "[v3-da3-b] $ds already complete"; return 0; }
  echo "[v3-da3-b] $ds START $(date '+%F %T') scenes:${scenes}"
  # shellcheck disable=SC2086
  $PY scripts/train_da3_protocol.py --dataset "$ds" --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h --teacher_N 8 --n_train 10 \
      --mask_ratio 0.5 --loss_all_pos $V2 \
      --swanlab --swanlab_suffix _v2full \
      >> "logs/v3_da3_${ds}.log" 2>&1 \
    && echo "[v3-da3-b] $ds DONE $(date '+%F %T')" \
    || echo "[v3-da3-b] $ds FAILED $(date '+%F %T')"
  $PY scripts/protocol_v2_analyze.py --model da3 --dataset "$ds" --run_dir "$out" \
      >> "logs/v3_da3_${ds}_analysis.log" 2>&1 || true
}

for ds in $DATASETS; do run_ds "$ds"; done
echo "[v3-da3-b] ALL DONE ($DATASETS) $(date '+%F %T')"
