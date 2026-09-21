#!/bin/bash
# Lane 2: hiroom then scannetpp
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/envs/da3/bin/python

run () {
  local ds=$1 scenes=$2
  local out=workspace/da3_protocol_${ds}_t16s4_lossall
  mkdir -p "$out"
  echo "[164] $ds START $(date '+%F %T')"
  $PY scripts/train_da3_protocol.py --dataset $ds --scenes $scenes \
      --output_root "$out" --steps 100 --arm rkdc1h \
      --teacher_N 16 --n_shared 4 --n_train 10 \
      --mask_ratio 0.5 --loss_all_pos \
      > logs/da3_${ds}_t16s4_lossall.log 2>&1 \
      || { echo "[164] $ds FAILED"; return 1; }
  echo "[164] $ds DONE $(date '+%F %T')"
}

HIROOM=$(tr '\n' ' ' < workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt)
run hiroom "$HIROOM"
run scannetpp "09c1414f1b 1ada7a0617 21d970d8de 286b55a2bf 38d58a7a31 3e8bba0176 40aec5fffa 578511c8a9 5f99900f09 7831862f02 7bc286c1b6 9071e139d9 acd95847c5 bcd2436daf bde1e479ad c4c04e6d6c c5439f4607 cc5237fd77 f3d64c30f8 fb5a96b1a2"
echo "[164-lane2] DONE $(date '+%F %T')"
