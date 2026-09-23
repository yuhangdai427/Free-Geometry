#!/bin/bash
# DA3 eth3d TTA v2: skip built-in eval (probe pipeline owns evaluation),
# skip scenes whose ckpt already exists. ckpts RETAINED.
set -u
cd /localhdd02/yuhang/code/free_geometry_latest
PY=/localhdd02/yuhang/envs/da3/bin/python
MAN=workspace/ndispatch/eth3d/scene_manifest.json
SCENES=$($PY -c "import json;print(' '.join(sorted(json.load(open('$MAN'))['scenes'])))")
SUM=workspace/overnight/eth3d_tta_SUMMARY.log
pat() { echo "[$(date '+%F %T')] $*" >> $SUM; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for SC in $SCENES; do
  OUT=workspace/overnight/da3_eth3d_u100/$SC
  CKPT=$OUT/ckpts/$SC/c2m_final_lora.pt
  if [ -f "$CKPT" ] || [ -d "$OUT/ckpts/$SC/c2m_final_lora_peft" ]; then
    pat "DA3 $SC OK (already trained, ckpt present)"
    continue
  fi
  pat "DA3 $SC start"
  if $PY scripts/train_pw0_accum.py --dataset eth3d --scene $SC --vggt_sync \
      --half_mode vggt --manifest $MAN --updates 100 --accum 1 --skip_eval \
      --out $OUT > logs/eth3d_da3_${SC}.log 2>&1; then
    pat "DA3 $SC OK (ckpt kept, eval skipped)"
  else
    pat "DA3 $SC FAILED"
  fi
done
pat "==== ETH3D DA3 TTA ALL DONE ===="
