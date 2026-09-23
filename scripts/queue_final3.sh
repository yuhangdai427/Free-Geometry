#!/bin/bash
# Three experiments on courtyard (first scene):
# 1. cwd_camtok raw, no camrel  — pure spatial CWD + camera token
# 2. cwd_camtok_ln post-LN, no camrel — same but on head_norm tokens
# 3. vggt_cwd raw + camrel — spatial CWD (no camtok) + decoded-pose camrel
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
SC=courtyard

echo "[exp1] cwd_camtok raw no-camrel $(date '+%F %T')"
$PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode cwd_camtok \
    --cwd_tau 0.5 --updates 100 --accum 1 \
    --out workspace/final3/ccamtok_raw/$SC > logs/final3_ccamtok_raw.log 2>&1
grep "cam_dec" logs/final3_ccamtok_raw.log
rm -rf workspace/final3/ccamtok_raw/$SC/ckpts workspace/final3/ccamtok_raw/$SC/recon

echo "[exp2] cwd_camtok_ln post-LN no-camrel $(date '+%F %T')"
$PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode cwd_camtok_ln \
    --cwd_tau 0.5 --updates 100 --accum 1 \
    --out workspace/final3/ccamtok_ln/$SC > logs/final3_ccamtok_ln.log 2>&1
grep "cam_dec" logs/final3_ccamtok_ln.log
rm -rf workspace/final3/ccamtok_ln/$SC/ckpts workspace/final3/ccamtok_ln/$SC/recon

echo "[exp3] vggt_cwd raw + camrel $(date '+%F %T')"
$PY scripts/train_pw0_accum.py --scene $SC --vggt_sync --half_mode vggt_cwd \
    --cwd_tau 0.5 --camrel --updates 100 --accum 1 \
    --out workspace/final3/cwd_camrel/$SC > logs/final3_cwd_camrel.log 2>&1
grep "cam_dec" logs/final3_cwd_camrel.log
rm -rf workspace/final3/cwd_camrel/$SC/ckpts workspace/final3/cwd_camrel/$SC/recon

echo "[final3] ALL DONE $(date '+%F %T')"
