#!/bin/bash
set -u
cd /root/autodl-tmp/Free-Geometry
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/da3/bin/python
DG=diagnostics/free_geometry
RR=workspace/protocol_v2/full16_vggt_hiroom
ARM=C2M_RKDC1H
VS=allv
$PY scripts/v2_materialize_selected.py --run_root "$RR" --arm $ARM --out_arm "${ARM}_SEL" \
    >> "logs/v3_vggt_hiroom.log" 2>&1 || exit 1
SKIP_BASELINE=1 $PY $DG/eval_viewcounts.py --manifest "$RR/scene_manifest.json" \
    --ckpt_root "$RR/ckpts_selected" --run_root "$RR" --step 100 \
    --arms "${ARM}_SEL" --view_subsets "$VS" >> "logs/v3_vggt_hiroom.log" 2>&1 || exit 1
$PY $DG/run_eval.py --run_root "$RR" --datas hiroom \
    --experiments "${ARM}_SEL@${VS}" >> "logs/v3_vggt_hiroom.log" 2>&1 || exit 1
mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_V2selected.json"
rm -rf "$RR"/eval32/*/model_results
echo "[vggt-hr-sel] DONE $(date '+%F %T')" >> logs/v3_vggt_hiroom.log
