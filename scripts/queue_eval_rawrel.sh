#!/bin/bash
# rawrel ckpts don't exist until fgraw finishes — wait ONLY for that, then
# evaluate (maskrel eval already ran in parallel earlier).
set -u
cd /root/autodl-tmp/Free-Geometry
for i in $(seq 1 300); do
  if ! pgrep -f "train_pw0_accum|train_arms.py" >/dev/null 2>&1 \
     && grep -q "all done" logs/queue_fgraw.log 2>/dev/null; then
    break
  fi
  sleep 20
done
echo "[evq2] eval VGGT C2M_rawrel ckpts $(date '+%F %T')"
/root/miniconda3/envs/da3/bin/python scripts/eval_vggt_cosw1.py \
    --scenes courtyard facade \
    --run_root workspace/vggt_rawrel_grad --arm C2M_rawrel --step 100 \
    > logs/eval_vggt_rawrel_grad.log 2>&1
echo "[evq2] rawrel eval rc=$? $(date '+%F %T')"
