#!/bin/bash
# eth3d full sweep: aligned / aligned_rkd / adaptive_s (11 scenes each)
# sim3 = Umeyama-aligned pose distillation (orientation-Procrustes R_a, 2026-09-20)
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python
SCENES="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"

run () {
  local arm=$1 out=$2
  mkdir -p "$out"
  echo "[${arm}] START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset eth3d --scenes $SCENES \
      --output_root "$out" --steps 100 --arm $arm --pose_weight 1.0 \
      --teacher_N 16 --n_shared 4 --n_train 10 --mask_ratio 0.5 --loss_all_pos \
      > logs/da3_${arm}_eth3d_full.log 2>&1 \
      || { echo "[${arm}] FAILED"; return 1; }
  echo "[${arm}] DONE $(date '+%F %T')"
}

case "${1:-}" in
  lane1) run aligned workspace/da3_aligned_eth3d ;;
  lane2) run aligned_rkd workspace/da3_alignedrkd_eth3d ;;
  lane3) run adaptive_s workspace/da3_adaptives_eth3d ;;
  lane4) run adaptive_a workspace/da3_adaptive_a_eth3d_cfix ;;
  *) echo "usage: $0 lane1|lane2|lane3|lane4" ;;
esac
