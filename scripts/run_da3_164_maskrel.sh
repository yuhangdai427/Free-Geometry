#!/bin/bash
# DA3 16:4 + all-position + multiview-only LoRA (13-39) + maskrel loss
# No baselines (baselines are frozen, never rerun)
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

run () {
  local ds=$1 scenes=$2
  local out=workspace/da3_164_maskrel_${ds}
  mkdir -p "$out"
  echo "[164-mr] $ds START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root "$out" --steps 100 --arm c2m --pose_weight 1.0 \
      --teacher_N 16 --n_shared 4 --n_train 10 \
      --mask_ratio 0.5 --loss_all_pos \
      > logs/da3_164_maskrel_${ds}.log 2>&1 \
      || { echo "[164-mr] $ds FAILED"; return 1; }
  echo "[164-mr] $ds DONE $(date '+%F %T')"
}

SCENES_7S="chess fire heads office pumpkin redkitchen stairs"
SCENES_ET="courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains"
SCENES_HR=$(tr '\n' ' ' < workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt)
SCENES_SP="09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"

# 4 datasets, 2 parallel lanes
(
  run 7scenes "$SCENES_7S"
  run eth3d   "$SCENES_ET"
) &
(
  run hiroom  "$SCENES_HR"
  run scannetpp "$SCENES_SP"
) &
wait
echo "[164-mr] ALL DONE $(date '+%F %T')"
