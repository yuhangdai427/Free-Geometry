#!/bin/bash
# DA3 F1/CD 回补: 7scenes + eth3d 的 baseline 与 rkdc1h（以及 eth3d C2M/pw0）
set -u
cd /root/autodl-tmp/Free-Geometry
OUT=artifacts/diagnostics/final_protocol/da3_baseline
LOG=$OUT/recon_backfill.log
M7=artifacts/diagnostics/final_protocol/7scenes/scene_manifest.json
ME=artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json
echo "=== recon backfill start $(date '+%F %T') ===" >> "$LOG"
python3 scripts/eval_da3_recon.py --dataset 7scenes --manifest "$M7" --arm A0_baseline \
    --out "$OUT/7scenes_recon_baseline.json" >> "$LOG" 2>&1 || { echo "7s baseline FAILED" >> "$LOG"; exit 1; }
python3 scripts/eval_da3_recon.py --dataset 7scenes --manifest "$M7" --arm rkdc1h \
    --ckpt_root workspace/da3_protocol_7scenes_rkdc1h/ckpts \
    --out "$OUT/7scenes_recon_rkdc1h.json" >> "$LOG" 2>&1 || { echo "7s rkdc1h FAILED" >> "$LOG"; exit 1; }
python3 scripts/eval_da3_recon.py --dataset eth3d --manifest "$ME" --arm A0_baseline \
    --out "$OUT/eth3d_recon_baseline.json" >> "$LOG" 2>&1 || { echo "e3 baseline FAILED" >> "$LOG"; exit 1; }
python3 scripts/eval_da3_recon.py --dataset eth3d --manifest "$ME" --arm rkdc1h \
    --ckpt_root workspace/da3_protocol_eth3d_rkdc1h/ckpts \
    --out "$OUT/eth3d_recon_rkdc1h.json" >> "$LOG" 2>&1 || { echo "e3 rkdc1h FAILED" >> "$LOG"; exit 1; }
echo "=== recon backfill DONE $(date '+%F %T') ===" >> "$LOG"
