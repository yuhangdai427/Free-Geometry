#!/bin/bash
# dev2 ABS 双臂对照：C2M_ABS(修复版) + C2M_ABSR(原始终靶) on phase4 开发场景对
set -euo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
RR=artifacts/diagnostics/final_protocol/scannetpp_dev2
LOG=$RR/stream_dev2_abs.log
mkdir -p "$RR"

# 1. mini manifest：逐字复制两个场景条目（同 train pairs、同 eval 帧）
python3 - << 'PYEOF'
import json
src = json.load(open('artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json'))
keep = ['7831862f02', 'bde1e479ad']
out = dict(src)
out['scenes'] = {k: src['scenes'][k] for k in keep}
out['run_root'] = 'artifacts/diagnostics/final_protocol/scannetpp_dev2'
out['note'] = 'dev2 ABS differential (7831862f02, bde1e479ad); entries verbatim from final_protocol/scannetpp'
json.dump(out, open('artifacts/diagnostics/final_protocol/scannetpp_dev2/scene_manifest.json', 'w'), indent=1)
print('manifest written:', list(out['scenes']))
PYEOF

echo "=== train $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$RR" --arms C2M_ABS C2M_ABSR \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1

echo "=== eval infer $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_ABS C2M_ABSR --view_subsets 100v >> "$LOG" 2>&1

echo "=== metrics $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/run_eval.py" --run_root "$RR" --datas scannetpp \
    --experiments "A0_baseline@100v" "C2M_ABS@100v" "C2M_ABSR@100v" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_ABS_dev2.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_dev2.csv" >> "$LOG" 2>&1
rm -rf "$RR"/eval32/*/model_results
echo "=== DONE $(date '+%F %T') ===" >> "$LOG"
echo "DEV2_ABS COMPLETE"
