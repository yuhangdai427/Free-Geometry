#!/usr/bin/env bash
set -euo pipefail

# DTU-49 follows the DA3-BENCH MVS reconstruction protocol.  It deliberately
# runs reconstruction with predicted poses only: DTU-49 is not a pose benchmark.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/workspace/benchmark_dataset/dtu}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/da3_dtu49_free_geometry}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu49}"
EVAL_SEED="${EVAL_SEED:-43}"
EVAL_FRAMES="${EVAL_FRAMES:-49}"

# Match the maintained DA3 ETH3D setup, with 10 actual samples per DTU scene.
TRAIN_SAMPLES_PER_SCENE="${TRAIN_SAMPLES_PER_SCENE:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-1e-4}"

cd "${REPO_ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }

check_dataset() {
    local required=(
        "${DATA_ROOT}/Rectified"
        "${DATA_ROOT}/Cameras"
        "${DATA_ROOT}/Points/stl"
        "${DATA_ROOT}/SampleSet/mvs_data/ObsMask"
    )
    local path
    for path in "${required[@]}"; do
        [[ -d "${path}" ]] || { echo "Missing DTU-49 asset: ${path}" >&2; return 1; }
    done

    local count
    count="$(find "${DATA_ROOT}/Rectified" -mindepth 1 -maxdepth 1 -type d -name 'scan*' | wc -l)"
    [[ "${count}" -eq 22 ]] || { echo "Expected 22 DTU-49 scenes, found ${count}" >&2; return 1; }
    echo "DTU-49 preflight passed: ${count} scenes, ${EVAL_FRAMES} evaluation views/scene."
}

verify_metrics() {
    local result_root="$1"
    "${PYTHON_BIN}" - "${result_root}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "metric_results" / "dtu_recon_unposed.json"
if not path.is_file():
    raise SystemExit(f"Missing reconstruction metrics: {path}")
metrics = json.loads(path.read_text())
scenes = {name: values for name, values in metrics.items() if name != "mean"}
if len(scenes) != 22:
    raise SystemExit(f"Expected 22 DTU scenes in {path}, found {len(scenes)}")
for name, values in scenes.items():
    for key in ("acc", "comp", "overall"):
        value = values.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f"Invalid {key} for {name}: {value!r}")
print(f"Verified DTU-49 reconstruction metrics: {path}")
PY
}

run_baseline() {
    local result_root="${WORKSPACE_ROOT}/baseline/seed${EVAL_SEED}"
    "${PYTHON_BIN}" - "${MODEL_NAME}" "${result_root}" "${EVAL_FRAMES}" "${EVAL_SEED}" <<'PY'
import sys
import torch

sys.path.insert(0, "src")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

model_name, work_dir, max_frames, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = DepthAnything3.from_pretrained(model_name).to(device)
evaluator = Evaluator(
    work_dir=work_dir,
    datas=["dtu"],
    modes=["recon_unposed"],
    max_frames=max_frames,
    seed=seed,
)
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
PY
    verify_metrics "${result_root}"
}

train_free_geometry() {
    "${PYTHON_BIN}" -u scripts/train_da3.py \
        --dataset dtu \
        --samples_per_scene "${TRAIN_SAMPLES_PER_SCENE}" \
        --model_name "${MODEL_NAME}" \
        --num_views 8 \
        --patch_huber_weight 1.0 --patch_huber_cos_weight -2.0 --patch_huber_delta 1.0 \
        --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256 \
        --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 \
        --cf_selection_mode mixed \
        --use_cf_distance --cf_distance_weight 1.0 --cf_distance_temperature 10.0 \
        --cf_distance_mode kl --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 \
        --epochs "${TRAIN_EPOCHS}" --batch_size 4 --num_workers 2 --lr "${TRAIN_LR}" \
        --lora_rank 32 --lora_alpha 32 \
        --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-7 --weight_decay 1e-5 \
        --output_dir "${OUTPUT_DIR}"
}

run_free_geometry() {
    local final_epoch=$((TRAIN_EPOCHS - 1))
    local lora_path="${OUTPUT_DIR}/epoch_${final_epoch}_lora.pt"
    local result_root="${WORKSPACE_ROOT}/free_geometry/epoch${final_epoch}_seed${EVAL_SEED}"
    [[ -f "${lora_path}" ]] || { echo "Missing Free-Geometry checkpoint: ${lora_path}" >&2; return 1; }

    "${PYTHON_BIN}" -u scripts/benchmark_da3.py \
        --lora_path "${lora_path}" --base_model "${MODEL_NAME}" \
        --lora_rank 32 --lora_alpha 32 \
        --datasets dtu --modes recon_unposed \
        --max_frames "${EVAL_FRAMES}" --seed "${EVAL_SEED}" \
        --work_dir "${result_root}"
    verify_metrics "${result_root}"
}

usage() {
    echo "Usage: $0 {check|baseline|train|free_geometry|all}" >&2
}

case "${1:-all}" in
    check) check_dataset ;;
    baseline) check_dataset; run_baseline ;;
    train) check_dataset; train_free_geometry ;;
    free_geometry) check_dataset; run_free_geometry ;;
    all) check_dataset; run_baseline; train_free_geometry; run_free_geometry ;;
    *) usage; exit 2 ;;
esac
