#!/bin/bash
# RAW-space patch loss (distance-dominant form, no LN/no conf) + UNCHANGED
# deployed rel-pose, both models, same manifest protocol — 4 component grads.
#   VGGT: train_arms.py --arms C2M_rawrel  (raw patchform + loss_pose_rel)
#   DA3 : train_pw0_accum.py --vggt_sync --half_mode vggt (raw + c2w rel)
# SERIAL chain: fgmgr2 -> fgk1 -> THIS (avoids two queues starting at once).
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry

for i in $(seq 1 400); do
  if ! pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgk1.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1; then
  echo "[raw] trainer STILL running — ABORT (no-parallel)"; exit 1
fi

mkdir -p workspace/vggt_rawrel_grad
cp artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json \
   workspace/vggt_rawrel_grad/scene_manifest.json

echo "[raw] VGGT C2M_rawrel grad run $(date '+%F %T')"
$PY $DG/train_arms.py --run_root workspace/vggt_rawrel_grad \
    --arms C2M_rawrel --scenes courtyard facade \
    --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components \
    > logs/vggt_rawrel_grad.log 2>&1
echo "[raw] VGGT done rc=$? $(date '+%F %T')"

for SC in courtyard facade; do
  echo "[raw] DA3 raw+rel $SC $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt --accum 1 \
      --updates 100 --out workspace/da3_rawrel_grad_$SC \
      > logs/da3_rawrel_grad_$SC.log 2>&1
  echo "[raw] DA3 $SC done rc=$? $(date '+%F %T')"
done
echo "[raw] all done $(date '+%F %T')"
