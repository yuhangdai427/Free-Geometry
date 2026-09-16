#!/usr/bin/env bash
set -euo pipefail

# Train one independent DA3 Free-Geometry LoRA per DTU-64 pose scene.
# The camera token remains trainable (the normal DA3 training behavior).
# Each scene is evaluated immediately, then large raw exports and intermediate
# checkpoints are pruned after the metric JSON has been validated.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu64_per_scene_lora_gpu1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/da3_dtu64_per_scene_lora_gpu1}"
GPU_ID="${GPU_ID:-1}"
EVAL_SEED="${EVAL_SEED:-43}"
TRAIN_SAMPLES_PER_SCENE="${TRAIN_SAMPLES_PER_SCENE:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-1e-4}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }

mapfile -t SCENES < <("${PYTHON_BIN}" - <<'PY'
import sys
sys.path.insert(0, "src")
from depth_anything_3.utils.constants import DTU64_SCENES

if len(DTU64_SCENES) != 13:
    raise SystemExit(f"Expected 13 DTU-64 scenes, found {len(DTU64_SCENES)}")
for scene in DTU64_SCENES:
    print(scene)
PY
)

verify_and_cleanup() {
    local scene="$1" result_root="$2" checkpoint_dir="$3"
    "${PYTHON_BIN}" - "${scene}" "${result_root}" <<'PY'
import json
import math
import sys
from pathlib import Path

scene, root = sys.argv[1:]
path = Path(root) / "metric_results/dtu64_pose.json"
data = json.loads(path.read_text())
values = data.get(scene)
if not isinstance(values, dict) or not all(math.isfinite(values.get(k, float("nan"))) for k in ("auc03", "auc30")):
    raise SystemExit(f"Invalid pose metrics for {scene}: {path}")
print(f"Verified {scene}: AUC@3={values['auc03']:.6f}, AUC@30={values['auc30']:.6f}")
PY

    local raw_dir="${result_root}/model_results"
    if [[ -d "${raw_dir}" ]]; then
        local bytes
        bytes="$(du -sb "${raw_dir}" | awk '{print $1}')"
        find "${raw_dir}" -depth -delete
        echo "Removed verified raw model results for ${scene}: ${bytes} bytes"
    fi

    # Retain only the final adapter and its camera-token sidecar for this scene.
    find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 \
        ! -name "epoch_$((TRAIN_EPOCHS - 1))_lora.pt" \
        ! -name "epoch_$((TRAIN_EPOCHS - 1))_lora_peft" -depth -exec rm -rf -- {} +
}

for scene in "${SCENES[@]}"; do
    final_epoch=$((TRAIN_EPOCHS - 1))
    checkpoint_dir="${CHECKPOINT_ROOT}/${scene}"
    result_root="${WORKSPACE_ROOT}/${scene}/free_geometry_epoch${final_epoch}_seed${EVAL_SEED}"
    lora_path="${checkpoint_dir}/epoch_${final_epoch}_lora.pt"

    echo "============================================================"
    echo "DTU-64 per-scene LoRA: ${scene} (camera token trainable)"
    echo "============================================================"
    "${PYTHON_BIN}" -u scripts/train_da3.py \
        --dataset dtu64 --scenes "${scene}" --samples_per_scene "${TRAIN_SAMPLES_PER_SCENE}" \
        --model_name "${MODEL_NAME}" --num_views 8 \
        --patch_huber_weight 1.0 --patch_huber_cos_weight -2.0 --patch_huber_delta 1.0 \
        --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256 \
        --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 --cf_selection_mode mixed \
        --use_cf_distance --cf_distance_weight 1.0 --cf_distance_temperature 10.0 --cf_distance_mode kl \
        --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 \
        --epochs "${TRAIN_EPOCHS}" --batch_size 4 --num_workers 2 --lr "${TRAIN_LR}" \
        --lora_rank 32 --lora_alpha 32 --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-7 \
        --weight_decay 1e-5 --log_interval 1 --output_dir "${checkpoint_dir}"

    [[ -f "${lora_path}" ]] || { echo "Missing trained LoRA: ${lora_path}" >&2; exit 1; }
    "${PYTHON_BIN}" -u scripts/benchmark_da3.py \
        --lora_path "${lora_path}" --base_model "${MODEL_NAME}" --lora_rank 32 --lora_alpha 32 \
        --datasets dtu64 --modes pose --scenes "${scene}" --max_frames 64 --seed "${EVAL_SEED}" \
        --work_dir "${result_root}"
    verify_and_cleanup "${scene}" "${result_root}" "${checkpoint_dir}"
done

"${PYTHON_BIN}" - "${WORKSPACE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*/free_geometry_epoch*_seed*/metric_results/dtu64_pose.json")):
    data = json.loads(path.read_text())
    summary.update({scene: values for scene, values in data.items() if scene != "mean"})
if len(summary) != 13:
    raise SystemExit(f"Expected 13 completed scene metrics, found {len(summary)}")
summary["mean"] = {
    key: sum(values[key] for scene, values in summary.items()) / len(summary)
    for key in ("auc03", "auc30")
}
out = root / "metric_results" / "dtu64_pose_per_scene_lora.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + "\n")
print(f"Saved per-scene LoRA summary: {out}")
print(f"Mean AUC@3={summary['mean']['auc03']:.6f}, AUC@30={summary['mean']['auc30']:.6f}")
PY
