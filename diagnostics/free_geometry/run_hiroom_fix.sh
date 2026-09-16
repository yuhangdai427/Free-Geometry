#!/bin/bash
# hiroom_fix: 重训 OOM 死掉的 12 场景（hiroom_fix manifest），ckpt 拷回主 hiroom run_root，
# 然后全 30 场景重推理覆盖被污染的 npz，重出指标覆盖被污染的 eval32_metrics_RKDC1.json
set -u
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
FIX=artifacts/diagnostics/final_protocol/hiroom_fix
RR=artifacts/diagnostics/final_protocol/hiroom
LOG="$FIX/fix_chain.log"
echo "=== hiroom_fix start $(date '+%F %T') ===" >> "$LOG"
python3 "$DG/train_arms.py" --run_root "$FIX" --arms C2M_RKDC1 \
    --epochs 10 --seed 0 --no_eval32 >> "$LOG" 2>&1 || { echo "TRAIN FAILED" >> "$LOG"; exit 1; }
# ckpt 拷回主 run_root
for d in "$FIX"/ckpts/20241230/*/*/; do
  rel=${d#"$FIX"/ckpts/}
  mkdir -p "$RR/ckpts/$rel"
  cp -r "$d"C2M_RKDC1 "$RR/ckpts/$rel"/ 2>> "$LOG"
done
echo "ckpts merged: $(find "$RR"/ckpts -name adapter_model.safetensors -path '*RKDC1*step100*' | wc -l)/30" >> "$LOG"
# 全量重推理（覆盖污染 npz）
SKIP_BASELINE=1 python3 "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
    --arms C2M_RKDC1 --view_subsets allv >> "$LOG" 2>&1
python3 "$DG/run_eval.py" --run_root "$RR" --datas hiroom \
    --experiments "C2M_RKDC1@allv" >> "$LOG" 2>&1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_RKDC1.json"
python3 "$DG/depth_metrics.py" --run_root "$RR" \
    --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_RKDC1.csv" >> "$LOG" 2>&1
echo "=== hiroom_fix COMPLETE $(date '+%F %T') ===" >> "$LOG"
