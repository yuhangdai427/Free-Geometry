#!/usr/bin/env bash
set -euo pipefail

# Full DA3 evaluation on both DTU protocols, serialized on physical GPU 1:
# DTU-64 pose first, then DTU-49 reconstruction.  Each protocol runs baseline,
# trains its own Free-Geometry LoRA, then evaluates that LoRA.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu_full_gpu1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/da3_dtu_full_gpu1}"
GPU_ID="${GPU_ID:-1}"
EVAL_SEED="${EVAL_SEED:-43}"

# Match the maintained ETH3D DA3 adaptation protocol.
TRAIN_SAMPLES_PER_SCENE="${TRAIN_SAMPLES_PER_SCENE:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-1e-4}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }

preflight() {
    "${PYTHON_BIN}" - <<'PY'
import sys
sys.path.insert(0, "src")
from depth_anything_3.bench.datasets.dtu import DTU
from depth_anything_3.bench.datasets.dtu64 import DTU64

for klass, expected_scenes, expected_views in ((DTU64, 13, 64), (DTU, 22, 49)):
    dataset = klass()
    if len(dataset.SCENES) != expected_scenes:
        raise SystemExit(f"{klass.__name__}: expected {expected_scenes} scenes, got {len(dataset.SCENES)}")
    data = dataset.get_data(dataset.SCENES[0])
    if len(data.image_files) != expected_views:
        raise SystemExit(f"{klass.__name__}: expected {expected_views} views, got {len(data.image_files)}")
    print(f"Preflight passed: {klass.__name__}: {expected_scenes} scenes x {expected_views} views")
PY
}

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
print(f"Verified {dataset} {mode}: {len(scenes)} scenes, {path}")
PY
}

run_baseline() {
    local dataset="$1" mode="$2" frames="$3" expected_scenes="$4"
    local result_root="${WORKSPACE_ROOT}/${dataset}/baseline_seed${EVAL_SEED}"
    "${PYTHON_BIN}" - "${MODEL_NAME}" "${dataset}" "${mode}" "${frames}" "${result_root}" "${EVAL_SEED}" <<'PY'
import sys
import torch
sys.path.insert(0, "src")
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator

model_name, dataset, mode, frames, root, seed = sys.argv[1:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = DepthAnything3.from_pretrained(model_name).to(device)
evaluator = Evaluator(work_dir=root, datas=[dataset], modes=[mode], max_frames=int(frames), seed=int(seed))
evaluator.infer(api)
metrics = evaluator.eval()
evaluator.print_metrics(metrics)
PY
    verify_metrics "${dataset}" "${mode}" "${expected_scenes}" "${result_root}"
}

train_and_evaluate_lora() {
    local dataset="$1" mode="$2" frames="$3" expected_scenes="$4"
    local checkpoint_dir="${CHECKPOINT_ROOT}/${dataset}"
    local final_epoch=$((TRAIN_EPOCHS - 1))
    local lora_path="${checkpoint_dir}/epoch_${final_epoch}_lora.pt"
    local result_root="${WORKSPACE_ROOT}/${dataset}/free_geometry_epoch${final_epoch}_seed${EVAL_SEED}"

    "${PYTHON_BIN}" -u scripts/train_da3.py \
        --dataset "${dataset}" --samples_per_scene "${TRAIN_SAMPLES_PER_SCENE}" \
        --model_name "${MODEL_NAME}" --num_views 8 \
        --patch_huber_weight 1.0 --patch_huber_cos_weight -2.0 --patch_huber_delta 1.0 \
        --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256 \
        --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 --cf_selection_mode mixed \
        --use_cf_distance --cf_distance_weight 1.0 --cf_distance_temperature 10.0 --cf_distance_mode kl \
        --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 \
        --epochs "${TRAIN_EPOCHS}" --batch_size 4 --num_workers 2 --lr "${TRAIN_LR}" \
        --lora_rank 32 --lora_alpha 32 --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-7 \
        --weight_decay 1e-5 --log_interval 1 --output_dir "${checkpoint_dir}"

    [[ -f "${lora_path}" ]] || { echo "Missing trained LoRA: ${lora_path}" >&2; return 1; }
    "${PYTHON_BIN}" -u scripts/benchmark_da3.py \
        --lora_path "${lora_path}" --base_model "${MODEL_NAME}" --lora_rank 32 --lora_alpha 32 \
        --datasets "${dataset}" --modes "${mode}" --max_frames "${frames}" --seed "${EVAL_SEED}" \
        --work_dir "${result_root}"
    verify_metrics "${dataset}" "${mode}" "${expected_scenes}" "${result_root}"
}

run_pose() {
    run_baseline dtu64 pose 64 13
    train_and_evaluate_lora dtu64 pose 64 13
}

run_reconstruction() {
    run_baseline dtu recon_unposed 49 22
    train_and_evaluate_lora dtu recon_unposed 49 22
}

case "${1:-all}" in
    check) preflight ;;
    pose) preflight; run_pose ;;
    recon) preflight; run_reconstruction ;;
    all) preflight; run_pose; run_reconstruction ;;
    *) echo "Usage: $0 {check|pose|recon|all}" >&2; exit 2 ;;
esac
