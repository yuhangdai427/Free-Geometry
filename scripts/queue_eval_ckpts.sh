#!/bin/bash
# Evaluate the saved grad-run ckpts (maskrel deployed + rawrel) after all
# training queues drain. Metric path identical to the DA3 evaluate_scene
# numbers reported all day (compute_pose AUC@3, LS-scale abs_rel, fuse3d F1).
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python

for i in $(seq 1 300); do
  if ! pgrep -f "train_pw0_accum|train_arms.py|eval_vggt_cosw1" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgraw.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "train_pw0_accum|train_arms.py|eval_vggt_cosw1" >/dev/null 2>&1; then
  echo "[evq] trainer/eval STILL running — ABORT (no-parallel)"; exit 1
fi

echo "[evq] eval VGGT C2M_maskrel ckpts $(date '+%F %T')"
$PY scripts/eval_vggt_cosw1.py --scenes courtyard facade \
    --run_root workspace/vggt_maskrel_grad --arm C2M_maskrel --step 100 \
    > logs/eval_vggt_maskrel_grad.log 2>&1
echo "[evq] maskrel eval rc=$? $(date '+%F %T')"

echo "[evq] eval VGGT C2M_rawrel ckpts $(date '+%F %T')"
$PY scripts/eval_vggt_cosw1.py --scenes courtyard facade \
    --run_root workspace/vggt_rawrel_grad --arm C2M_rawrel --step 100 \
    > logs/eval_vggt_rawrel_grad.log 2>&1
echo "[evq] rawrel eval rc=$? $(date '+%F %T')"
echo "[evq] all done $(date '+%F %T')"
