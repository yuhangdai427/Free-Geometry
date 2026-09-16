#!/bin/bash
# 24:8 + RKDC1：双病灶场景（21d970d8de 融合漂移 + 1ada7a0617 量规脱钩）
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_248_rkdc1
LOG=$RR/stream_248.log
mkdir -p "$RR"

# 24:8 manifest（复用 build_dyn_manifest.build，eval 帧与主 manifest 逐字一致）
python3 - << 'PYEOF'
import json, sys, os
sys.path.insert(0, 'diagnostics/free_geometry')
from build_dyn_manifest import build
ref = json.load(open('artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json'))
scenes = ['21d970d8de', '1ada7a0617']
out = {"dataset": "scannetpp", "run_root": "artifacts/diagnostics/final_protocol/scannetpp_248_rkdc1",
       "note": "24:8 + RKDC1, drift(21d97)+gauge(1ada) scenes", "scenes": {}}
for sc in scenes:
    ev = ref["scenes"][sc]["eval32_frames"]
    out["scenes"][sc] = build("scannetpp", sc, 24, 8, "random", 10, ev)
json.dump(out, open(f'{out["run_root"]}/scene_manifest.json', 'w'), indent=1)
print('manifest written:', list(out['scenes']))
PYEOF

# 等 PMC 收尾
while pgrep -f "train_arms.py|eval_viewcounts.py|run_eval.py" > /dev/null; do sleep 30; done

echo "=== train $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1 \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1

echo "=== eval infer $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1 --view_subsets 100v >> "$LOG" 2>&1

echo "=== metrics $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "A0_baseline@100v" "C2M_RKDC1@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_248.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_248.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results
echo "=== DONE $(date '+%F %T') ===" >> "$LOG"
echo "248_RKDC1 COMPLETE"
