#!/bin/bash
# Parallel dtu64 lane: runs dtu64 concurrently with the main lane's dtu phase.
# Cooperates via the same output dir + step100-skip semantics of the main lane.
# No --swanlab (avoid run spam); writes to the same log with a marker.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
MAN=workspace/protocol_v2/ab_manifests
V2="--v2_ab_manifest $MAN --v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_qfeat_off --v2_rel_gate_deg 0 --v2_apply_selection --v2_baselines_json workspace/protocol_v2/baselines.json"
OUT=workspace/protocol_v2/full_da3_dtu64
LOG=logs/v3_da3_dtu64_par.log

scenes=$($PY - <<'EOF'
import glob, json
for p in sorted(glob.glob("workspace/protocol_v2/ab_manifests/dtu64/*.json")):
    try:
        print(json.load(open(p))["scene"])
    except Exception:
        pass
EOF
)

echo "[v3-da3-par] dtu64 START $(date '+%F %T') scenes: $scenes" >> "$LOG"
# shellcheck disable=SC2086
$PY scripts/train_da3_protocol.py --dataset dtu64 --scenes $scenes \
    --output_root "$OUT" --steps 100 --arm rkdc1h --teacher_N 8 --n_train 10 \
    --mask_ratio 0.5 --loss_all_pos $V2 \
    >> "$LOG" 2>&1 \
  && echo "[v3-da3-par] dtu64 DONE $(date '+%F %T')" >> "$LOG" \
  || echo "[v3-da3-par] dtu64 FAILED $(date '+%F %T')" >> "$LOG"
