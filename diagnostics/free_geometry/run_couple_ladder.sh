#!/bin/bash
# couple ladder: dev3 × {C2M_RKDC2 (couple 2.0), C2M_RKDC3 (couple 3.0)} — 验证 F1 对 couple 权重是否继续单调
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_dev3
LOG="$RR/ladder_chain.log"
echo "=== ladder start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC2 C2M_RKDC3 \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC2 C2M_RKDC3 --view_subsets 100v >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "C2M_RKDC2@100v" "C2M_RKDC3@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC23.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_dev3_rkdc23.csv" >> "$LOG" 2>&1
echo "=== ladder COMPLETE $(date '+%F %T') ===" >> "$LOG"
