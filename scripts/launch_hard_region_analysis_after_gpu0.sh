#!/usr/bin/env bash
set -euo pipefail

# Wait for the active 7Scenes/HiRoom queue before allocating VGGT models.
# A smoke result must contain finite, nonzero masked metrics before full runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PYTHON_BIN=/root/miniconda3/envs/da3/bin/python
MODEL="$ROOT/model_weights/VGGT-1B"
ART_MAIN="$ROOT/artifacts/hard_view_subsets"
ART_EXTRA="$ROOT/artifacts/hard_view_subsets_7scenes_hiroom"
OUT="$ROOT/results/hard_region_analysis"

while tmux has-session -t hard_view_7scenes_hiroom_gpu0 2>/dev/null; do sleep 30; done

run_region() {
  local ds="$1" criterion="$2" output="$3" windows="${4:-25}" art="$ART_MAIN" ckpt="$ROOT/checkpoints/hard_view_subsets" selection lora
  if [[ "$ds" == "7scenes" || "$ds" == "hiroom" ]]; then
    art="$ART_EXTRA"
    ckpt="$ROOT/checkpoints/hard_view_subsets_7scenes_hiroom"
    selection="$art/$ds/source_frames.jsonl"
    lora="$ckpt/$ds/latest_lora.pt"
  else
    selection="$art/$ds/$criterion/source_frames.jsonl"
    lora="$ckpt/$ds/$criterion/latest_lora.pt"
  fi
  "$PYTHON_BIN" scripts/evaluate_hard_regions.py --dataset "$ds" --criterion "$criterion" \
    --selection "$selection" --manifest "$art/$ds/$criterion/eval.jsonl" \
    --selection-report "$art/$ds/$criterion/selection_report.json" \
    --lora "$lora" \
    --model-name "$MODEL" --output "$output" --max-windows "$windows" --device cuda:0
}

mkdir -p "$OUT/smoke"
run_region eth3d texture "$OUT/smoke/eth3d_texture.json" 1
run_region scannetpp occlusion "$OUT/smoke/scannetpp_occlusion.json" 1
run_region 7scenes texture "$OUT/smoke/7scenes_texture.json" 1
run_region hiroom occlusion "$OUT/smoke/hiroom_occlusion.json" 1
"$PYTHON_BIN" - "$OUT/smoke/eth3d_texture.json" "$OUT/smoke/scannetpp_occlusion.json" "$OUT/smoke/7scenes_texture.json" "$OUT/smoke/hiroom_occlusion.json" <<'PY'
import json, math, sys
for path in sys.argv[1:]:
    data = json.load(open(path))
    for arm, metrics in data['mean'].items():
        for key in ('masked_absrel', 'overall', 'fscore', 'masked_pixels'):
            value = metrics[key]
            if not math.isfinite(value) or value <= 0:
                raise SystemExit(f'{path}: invalid {arm}/{key}={value}')
print('[smoke passed] region metrics are finite and nonzero')
PY

for ds in eth3d scannetpp 7scenes hiroom; do
  for criterion in texture occlusion; do
    run_region "$ds" "$criterion" "$OUT/${ds}_${criterion}.json"
  done
done
echo "[done] $(date -u +%FT%TZ)"
