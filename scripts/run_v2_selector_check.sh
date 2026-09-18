#!/bin/bash
# Selector verification: for facade (picked step60) and electro (fell back to
# step0), evaluate several candidate steps on the real benchmark and check
# whether the selector's choice is actually right/better.
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
SRC=workspace/protocol_v2/vggt_eth3d
CHK=workspace/protocol_v2/selector_check
LOG=logs/v2_selector_check.log

$PY - <<'EOF'
import json, os
m = json.load(open(f'workspace/protocol_v2/vggt_eth3d/scene_manifest.json'))
m["scenes"] = {k: v for k, v in m["scenes"].items() if k in ("facade", "electro")}
os.makedirs('workspace/protocol_v2/selector_check', exist_ok=True)
json.dump(m, open('workspace/protocol_v2/selector_check/scene_manifest.json', 'w'), indent=1)
print("manifest scenes:", list(m["scenes"]))
EOF

for S in 60 100; do
  R=$CHK/s$S
  mkdir -p "$R"
  for sc in facade electro; do
    dst=$R/ckpts/$sc/C2M_RKDC1H
    mkdir -p "$dst"
    rm -rf "$dst/step100_lora_peft"
    cp -r "$SRC/ckpts/$sc/C2M_RKDC1H/v2/step${S}_lora_peft" "$dst/step100_lora_peft"
  done
  echo "[sel-check] step $S eval START $(date '+%F %T')"
  SKIP_BASELINE=1 $PY $DG/eval_viewcounts.py --manifest $CHK/scene_manifest.json \
      --ckpt_root "$R/ckpts" --run_root "$R" --step 100 \
      --arms C2M_RKDC1H --view_subsets allv >> "$LOG" 2>&1 || { echo "[sel-check] step $S FAILED"; continue; }
  $PY $DG/run_eval.py --run_root "$R" --datas eth3d --manifest $CHK/scene_manifest.json \
      --experiments "C2M_RKDC1H@allv" >> "$LOG" 2>&1 || continue
  mv "$R/eval32_metrics.json" "$R/metrics.json"
  rm -rf "$R"/eval32/*/model_results
  echo "[sel-check] step $S DONE $(date '+%F %T')"
done
echo "SELECTOR_CHECK_ALL_DONE"
