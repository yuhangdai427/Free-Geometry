#!/bin/bash
# Raymap-camera eval (seeded) for the two 16:4 cosw1 ckpts — SERIAL, no parallel.
# Each call evaluates BOTH the zero-LoRA baseline and the TTA ckpt with
# use_ray_pose=True (cameras+intrinsics from the ray map), incl. recon F1.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

if pgrep -f "eval_da3_raypose.py|train_pw0_accum.py" >/dev/null 2>&1; then
  echo "[rayq] another trainer/eval is running — ABORT (no-parallel)"; exit 1
fi

echo "[rayq] courtyard 16:4 raypose eval $(date '+%F %T')"
$PY scripts/eval_da3_raypose.py --dataset eth3d --scene courtyard \
    --run_root workspace/cosw1_grad_courtyard_t16 --seed 0 \
    > logs/raypose16_courtyard.log 2>&1
echo "[rayq] courtyard done $(date '+%F %T')"

echo "[rayq] facade 16:4 raypose eval $(date '+%F %T')"
$PY scripts/eval_da3_raypose.py --dataset eth3d --scene facade \
    --run_root workspace/cosw1_grad_facade --seed 0 \
    > logs/raypose16_facade.log 2>&1
echo "[rayq] facade done $(date '+%F %T')"
