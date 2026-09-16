#!/bin/bash
# dev3b：场景集 {7831862f02, 21d970d8de(融合漂移型), 1ada7a0617}
# 双臂：C2M_RKDC1（冠军，补 21d97 空白）+ C2M_RKDCX（RKDC1 + 0.3·xac2）
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_dev3b
LOG=$RR/stream_dev3b.log
mkdir -p "$RR"

python3 - << 'PYEOF'
import json
src = json.load(open('artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json'))
keep = ['7831862f02', '21d970d8de', '1ada7a0617']
out = dict(src)
out['scenes'] = {k: src['scenes'][k] for k in keep}
out['run_root'] = 'artifacts/diagnostics/final_protocol/scannetpp_dev3b'
out['note'] = 'dev3b: RKDC1/RKDCX on {783, 21d97(fusion-drift), 1ada}; entries verbatim'
json.dump(out, open('artifacts/diagnostics/final_protocol/scannetpp_dev3b/scene_manifest.json', 'w'), indent=1)
print('manifest written:', list(out['scenes']))
PYEOF

echo "=== train $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_RKDC1 C2M_RKDCX \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1

echo "=== eval infer $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1 C2M_RKDCX --view_subsets 100v >> "$LOG" 2>&1

echo "=== metrics $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "A0_baseline@100v" "C2M_RKDC1@100v" "C2M_RKDCX@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1X_dev3b.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_dev3b.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results
echo "=== DONE $(date '+%F %T') ===" >> "$LOG"
echo "DEV3B COMPLETE"
