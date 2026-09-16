#!/bin/bash
# dev3：C2M_XRKD（rel-pose + 绝对平移Huber+FoV，权重1.0）三场景对照
# 场景：7831862f02(强) / bde1e479ad(中) / 1ada7a0617(C2M最差)
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_dev3
LOG=$RR/stream_dev3_xrkd.log
mkdir -p "$RR"

python3 - << 'PYEOF'
import json
src = json.load(open('artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json'))
keep = ['7831862f02', 'bde1e479ad', '1ada7a0617']
out = dict(src)
out['scenes'] = {k: src['scenes'][k] for k in keep}
out['run_root'] = 'artifacts/diagnostics/final_protocol/scannetpp_dev3'
out['note'] = 'dev3 C2M_XRKD (rel + absT/FL w=1.0); entries verbatim from final_protocol/scannetpp'
json.dump(out, open('artifacts/diagnostics/final_protocol/scannetpp_dev3/scene_manifest.json', 'w'), indent=1)
print('manifest written:', list(out['scenes']))
PYEOF

echo "=== train $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_XRKD \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1

echo "=== eval infer $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_XRKD --view_subsets 100v >> "$LOG" 2>&1

echo "=== metrics $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "A0_baseline@100v" "C2M_XRKD@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_XRKD_dev3.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_dev3.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results
echo "=== DONE $(date '+%F %T') ===" >> "$LOG"
echo "DEV3_XRKD COMPLETE"
