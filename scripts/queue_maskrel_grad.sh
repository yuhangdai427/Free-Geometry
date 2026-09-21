#!/bin/bash
# Deployed-recipe gradient comparison: VGGT C2M_maskrel via its OWN trainer
# (train_arms.py), then DA3 conformed (--vggt_sync: same manifest pairs, same
# geometry/masks, deployed B5 loss + rel-pose). 4 component grads each.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry

if pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1; then
  echo "[mgr] another trainer running — ABORT (no-parallel)"; exit 1
fi

echo "[mgr] VGGT C2M_maskrel grad run $(date '+%F %T')"
$PY $DG/train_arms.py --run_root workspace/vggt_maskrel_grad \
    --arms C2M_maskrel --scenes courtyard facade \
    --epochs 10 --seed 0 --no_eval32 --loss_all_pos --grad_components \
    > logs/vggt_maskrel_grad.log 2>&1
echo "[mgr] VGGT done rc=$? $(date '+%F %T')"

for SC in courtyard facade; do
  echo "[mgr] DA3 vggt_sync $SC $(date '+%F %T')"
  $PY scripts/train_pw0_accum.py --scene $SC --vggt_sync \
      --loss_form huber --half_mode joint --updates 100 \
      --out workspace/da3_maskrel_grad_$SC \
      > logs/da3_maskrel_grad_$SC.log 2>&1
  echo "[mgr] DA3 $SC done rc=$? $(date '+%F %T')"
done
echo "[mgr] all done $(date '+%F %T')"
