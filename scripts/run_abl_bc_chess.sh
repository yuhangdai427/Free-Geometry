#!/bin/bash
# 7scenes/chess 分诊消融（串行，等显存）：隔离 v2ab 对 / couple_fix / rel 三个变量
# 全部冻结副本，mv13 参考 = 在线对+旧couple+rel off（+12%/+6%）
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
FR=/root/autodl-tmp/fg_abl/scripts/train_da3_protocol.py

wait_gpu () {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
    [ "${used:-99999}" -lt 55000 ] && break
    echo "[ablBC] waiting GPU (${used}MiB) $(date '+%T')"; sleep 60
  done
}

# B: 在线对 + 全机制（rel on, couple_fix on）——隔离"选帧对"
wait_gpu
$PY $FR --dataset 7scenes --scenes chess --output_root workspace/ablB_chess_online \
  --steps 100 --arm rkdc1h --teacher_N 8 --n_train 10 --mask_ratio 0.5 --loss_all_pos \
  --v2_probe --v2_ckpt --v2_rel_weight 1.0 --v2_grad_cap --v2_couple_fix \
  >> logs/ablB_chess.log 2>&1 && echo "[ablB] chess DONE"

# C: v2ab 对 + rel off + couple_fix OFF（旧非对称 couple）——隔离 couple_fix
wait_gpu
$PY $FR --dataset 7scenes --scenes chess --output_root workspace/ablC_chess_oldcouple \
  --steps 100 --arm rkdc1h --teacher_N 8 --n_train 10 --mask_ratio 0.5 --loss_all_pos \
  --v2_ab_manifest workspace/protocol_v2/ab_manifests \
  --v2_probe --v2_ckpt --v2_rel_weight 0.0 \
  >> logs/ablC_chess.log 2>&1 && echo "[ablC] chess DONE"
echo "ABL_BC_ALL_DONE"
