#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate the completed full-run baseline and LoRA at a 32-view cap.
# No training is performed; each scene receives the same deterministic seed-43
# sampling for baseline and LoRA so their metrics are directly comparable.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/da3_dtu_full_gpu1}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu_eval32_gpu1}"
GPU_ID="${GPU_ID:-1}"
EVAL_SEED="${EVAL_SEED:-43}"
MAX_FRAMES="${MAX_FRAMES:-32}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ "${MAX_FRAMES}" -eq 32 ]] || { echo "This runner is fixed to MAX_FRAMES=32" >&2; exit 2; }

verify_metrics() {
    local dataset="$1" mode="$2" expected_scenes="$3" root="$4"
    "${PYTHON_BIN}" - "${dataset}" "${mode}" "${expected_scenes}" "${root}" <<'PY'
import json
import math
import sys
from pathlib import Path

dataset, mode, expected_scenes, root = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
path = root / "metric_results" / f"{dataset}_{mode}.json"
if not path.is_file():
    raise SystemExit(f"Missing metrics: {path}")
data = json.loads(path.read_text())
scenes = {key: value for key, value in data.items() if key != "mean"}
if len(scenes) != expected_scenes:
    raise SystemExit(f"Expected {expected_scenes} scene results in {path}, got {len(scenes)}")
keys = ("auc03", "auc30") if mode == "pose" else ("acc", "comp", "overall")
for scene, result in scenes.items():
    for key in keys:
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f"Invalid {dataset}/{scene} {key}: {value!r}")
print(f"Verified {dataset} {mode} at {32} views: {path}")
PY
}

cleanup_raw_results() {
    local root="$1"
    local raw_dir="${root}/model_results"
    [[ -d "${raw_dir}" ]] || return 0
    local before_bytes
    before_bytes="$(du -sb "${raw_dir}" | awk '{print $1}')"
    rm -rf "${raw_dir}"
    echo "Removed verified raw model results: ${raw_dir} (${before_bytes} bytes)"
}

run_baseline() {
    local dataset="$1" mode="$2" expected_scenes="$3"
    local root="${WORKSPACE_ROOT}/${dataset}/baseline_seed${EVAL_SEED}_32v"
    "${PYTHON_BIN}" - "${MODEL_NAME}" "${dataset}" "${mode}" "${root}" "${EVAL_SEED}" "${MAX_FRAMES}" <<'PY'
import sys
import torch
sys.path.insert(0, "src")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

model_name, dataset, mode, root, seed, max_frames = sys.argv[1:]
api = DepthAnything3.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
evaluator = Evaluator(work_dir=root, datas=[dataset], modes=[mode], max_frames=int(max_frames), seed=int(seed))
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
PY
    verify_metrics "${dataset}" "${mode}" "${expected_scenes}" "${root}"
    cleanup_raw_results "${root}"
}

run_lora() {
    local dataset="$1" mode="$2" expected_scenes="$3"
    local lora_path="${SOURCE_CHECKPOINT_ROOT}/${dataset}/epoch_2_lora.pt"
    local root="${WORKSPACE_ROOT}/${dataset}/lora_epoch2_seed${EVAL_SEED}_32v"
    [[ -f "${lora_path}" ]] || { echo "Missing source LoRA checkpoint: ${lora_path}" >&2; return 1; }
    "${PYTHON_BIN}" -u scripts/benchmark_da3.py \
        --lora_path "${lora_path}" --base_model "${MODEL_NAME}" --lora_rank 32 --lora_alpha 32 \
        --datasets "${dataset}" --modes "${mode}" --max_frames "${MAX_FRAMES}" --seed "${EVAL_SEED}" \
        --work_dir "${root}"
    verify_metrics "${dataset}" "${mode}" "${expected_scenes}" "${root}"
    cleanup_raw_results "${root}"
}

run_pose() {
    run_baseline dtu64 pose 13
    run_lora dtu64 pose 13
}

run_reconstruction() {
    run_baseline dtu recon_unposed 22
    run_lora dtu recon_unposed 22
}

case "${1:-all}" in
    pose) run_pose ;;
    recon) run_reconstruction ;;
    all) run_pose; run_reconstruction ;;
    *) echo "Usage: $0 {pose|recon|all}" >&2; exit 2 ;;
esac
