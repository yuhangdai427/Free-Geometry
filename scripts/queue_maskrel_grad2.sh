#!/bin/bash
# CORRECTED deployed-recipe gradient comparison (v2):
#  - VGGT C2M_maskrel via train_arms.py, with scene_manifest.json copied into
#    the run_root (that was the earlier failure)
#  - DA3 --vggt_sync with FIXED code: rel-pose on c2w (convention-identical
#    to loss_pose_rel) + eval_frames from the manifest (proto bug fix)
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry

# wait for the previous queue to drain (its facade run will crash at eval)
for i in $(seq 1 120); do
  if ! pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgmgr.log 2>/dev/null; then
    break
  fi
  sleep 20
done
if pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1; then
  echo "[mgr2] trainer STILL running — ABORT (no-parallel)"; exit 1
fi

mkdir -p workspace/vggt_maskrel_grad
cp artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json \
   workspace/vggt_maskrel_grad/scene_manifest.json

echo "[mgr2] VGGT C2M_maskrel grad run $(date '+%F %T')"
$PY $DG/train_arms.py --run_root workspace/vggt_maskrel_grad \
    --arms C2M_maskrel --scenes courtyard facade \
    --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components \
    > logs/vggt_maskrel_grad2.log 2>&1
echo "[mgr2] VGGT done rc=$? $(date '+%F %T')"

for SC in courtyard facade; do
  echo "[mgr2] DA3 vggt_sync(c2w) $SC $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync \
      --loss_form huber --half_mode joint --updates 100 \
      --out workspace/da3_maskrel2_grad_$SC \
      > logs/da3_maskrel2_grad_$SC.log 2>&1
  echo "[mgr2] DA3 $SC done rc=$? $(date '+%F %T')"
done
echo "[mgr2] all done $(date '+%F %T')"
