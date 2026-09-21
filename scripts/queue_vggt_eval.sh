#!/bin/bash
# Evaluate VGGT cosw1 nt5 ckpts + zero-LoRA baseline (courtyard, facade).
set -u
cd /root/autodl-tmp/Free-Geometry
if pgrep -f "train_pw0_accum|train_vggt_pw0_accum|eval_da3_raypose|eval_vggt_cosw1" >/dev/null 2>&1; then
  echo "[vge] another job running — ABORT (no-parallel)"; exit 1
fi
echo "[vge] start $(date '+%F %T')"
/root/miniconda3/envs/da3/bin/python scripts/eval_vggt_cosw1.py \
    > logs/vggt_cosw1_eval.log 2>&1
echo "[vge] done rc=$? $(date '+%F %T')"
