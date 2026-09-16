#!/usr/bin/env bash
set -euo pipefail

# Sequential GPU-1 smoke validation: DTU-64 pose first, then DTU-49 reconstruction.
# Each dataset gets an independent baseline, a scene-specific Free-Geometry LoRA,
# and a LoRA evaluation over the same complete scene.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu_smoke_gpu1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/da3_dtu_smoke_gpu1}"
GPU_ID="${GPU_ID:-1}"
EVAL_SEED="${EVAL_SEED:-43}"
TRAIN_SAMPLES_PER_SCENE="${TRAIN_SAMPLES_PER_SCENE:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"

# One representative scene per protocol.  Keep all native views for evaluation.
DTU64_SCENE="${DTU64_SCENE:-scan105}"
DTU49_SCENE="${DTU49_SCENE:-scan1}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }

preflight() {
    "${PYTHON_BIN}" - "${DTU64_SCENE}" "${DTU49_SCENE}" <<'PY'
import sys
sys.path.insert(0, "src")
from depth_anything_3.bench.datasets.dtu import DTU
from depth_anything_3.bench.datasets.dtu64 import DTU64

for klass, scene, expected in ((DTU64, sys.argv[1], 64), (DTU, sys.argv[2], 49)):
    dataset = klass()
    data = dataset.get_data(scene)
    if len(data.image_files) != expected:
        raise SystemExit(f"{klass.__name__}/{scene}: expected {expected} views, found {len(data.image_files)}")
    print(f"Preflight passed: {klass.__name__}/{scene}, {expected} views")
PY
}

verify() {
    local dataset="$1" mode="$2" root="$3" scene="$4"
    "${PYTHON_BIN}" - "${dataset}" "${mode}" "${root}" "${scene}" <<'PY'
import json
import math
import sys
import time
import zipfile
from pathlib import Path

dataset, mode, root, scene = sys.argv[1:]
path = Path(root) / "metric_results" / f"{dataset}_{mode}.json"
if not path.is_file():
    raise SystemExit(f"Missing metric file: {path}")
data = json.loads(path.read_text())
values = data.get(scene)
if not isinstance(values, dict):
    raise SystemExit(f"Missing scene metrics for {scene} in {path}")
keys = ("auc03", "auc30") if mode == "pose" else ("acc", "comp", "overall")
for key in keys:
    value = values.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"Invalid {key} in {path}: {value!r}")
print(f"Verified {dataset}/{scene} {mode}: {path}")
PY
}

baseline() {
    local dataset="$1" mode="$2" scene="$3" frames="$4" root="$5"
    "${PYTHON_BIN}" - "${MODEL_NAME}" "${dataset}" "${mode}" "${scene}" "${frames}" "${root}" "${EVAL_SEED}" <<'PY'
import sys
import time
import zipfile
import torch
sys.path.insert(0, "src")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

model_name, dataset, mode, scene, frames, root, seed = sys.argv[1:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = DepthAnything3.from_pretrained(model_name).to(device)
evaluator = Evaluator(work_dir=root, datas=[dataset], modes=[mode], scenes=[scene], max_frames=int(frames), seed=int(seed))
evaluator.infer(api)
result_path = evaluator._export_dir(dataset, scene, posed=False) + "/exports/mini_npz/results.npz"
for attempt in range(30):
    try:
        with zipfile.ZipFile(result_path) as archive:
            if archive.testzip() is not None:
                raise zipfile.BadZipFile("corrupt member")
        break
    except (FileNotFoundError, EOFError, OSError, zipfile.BadZipFile):
        if attempt == 29:
            raise RuntimeError(f"Timed out waiting for a complete export: {result_path}")
        time.sleep(1)
for attempt in range(3):
    metrics = evaluator.eval()
    if f"{dataset}_{mode}" in metrics:
        break
    if attempt == 2:
        raise RuntimeError(f"Evaluation did not produce {dataset}_{mode} after inference")
    print("Result export was not visible yet; retrying evaluation in 2 seconds.")
    time.sleep(2)
evaluator.print_metrics(metrics)
PY
    verify "${dataset}" "${mode}" "${root}" "${scene}"
}

train_and_evaluate_lora() {
    local dataset="$1" mode="$2" scene="$3" frames="$4" lr="$5"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${dataset}_${scene}"
    local final_epoch=$((TRAIN_EPOCHS - 1))
    local lora_path="${checkpoint_dir}/epoch_${final_epoch}_lora.pt"
    local result_root="${WORKSPACE_ROOT}/${dataset}_${scene}/free_geometry_epoch${final_epoch}"

    "${PYTHON_BIN}" -u scripts/train_da3.py \
        --dataset "${dataset}" --scenes "${scene}" --samples_per_scene "${TRAIN_SAMPLES_PER_SCENE}" \
        --model_name "${MODEL_NAME}" --num_views 8 \
        --patch_huber_weight 1.0 --patch_huber_cos_weight -2.0 --patch_huber_delta 1.0 \
        --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256 \
        --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 --cf_selection_mode mixed \
        --use_cf_distance --cf_distance_weight 1.0 --cf_distance_temperature 10.0 --cf_distance_mode kl \
        --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 \
        --epochs "${TRAIN_EPOCHS}" --batch_size 4 --num_workers 2 --lr "${lr}" \
        --lora_rank 32 --lora_alpha 32 --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-7 \
        --weight_decay 1e-5 --log_interval 1 --output_dir "${checkpoint_dir}"

    [[ -f "${lora_path}" ]] || { echo "Missing trained LoRA: ${lora_path}" >&2; return 1; }
    "${PYTHON_BIN}" -u scripts/benchmark_da3.py \
        --lora_path "${lora_path}" --base_model "${MODEL_NAME}" --lora_rank 32 --lora_alpha 32 \
        --datasets "${dataset}" --modes "${mode}" --scenes "${scene}" --max_frames "${frames}" \
        --seed "${EVAL_SEED}" --work_dir "${result_root}"
    verify "${dataset}" "${mode}" "${result_root}" "${scene}"
}

run_pose() {
    local baseline_root="${WORKSPACE_ROOT}/dtu64_${DTU64_SCENE}/baseline"
    baseline dtu64 pose "${DTU64_SCENE}" 64 "${baseline_root}"
    train_and_evaluate_lora dtu64 pose "${DTU64_SCENE}" 64 1e-4
}

run_reconstruction() {
    local baseline_root="${WORKSPACE_ROOT}/dtu_${DTU49_SCENE}/baseline"
    baseline dtu recon_unposed "${DTU49_SCENE}" 49 "${baseline_root}"
    train_and_evaluate_lora dtu recon_unposed "${DTU49_SCENE}" 49 1e-4
}

case "${1:-all}" in
    check) preflight ;;
    pose) preflight; run_pose ;;
    recon) preflight; run_reconstruction ;;
    all) preflight; run_pose; run_reconstruction ;;
    *) echo "Usage: $0 {check|pose|recon|all}" >&2; exit 2 ;;
esac
