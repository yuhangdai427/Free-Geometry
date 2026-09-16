#!/usr/bin/env bash
set -euo pipefail

# Train one independent DA3 Free-Geometry LoRA per DTU-49 scene, then evaluate
# reconstruction with its predicted poses.  The camera token stays trainable.
# Existing full-view baseline metrics are reused; this runner does not rerun them.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-depth-anything/DA3-GIANT-1.1}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/da3_dtu49_per_scene_lora_gpu1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/da3_dtu49_per_scene_lora_gpu1}"
BASELINE_METRICS="${BASELINE_METRICS:-${REPO_ROOT}/workspace/da3_dtu_full_gpu1/dtu/baseline_seed43/metric_results/dtu_recon_unposed.json}"
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
[[ -f "${BASELINE_METRICS}" ]] || { echo "Missing full-view baseline metrics: ${BASELINE_METRICS}" >&2; exit 1; }

mapfile -t SCENES < <("${PYTHON_BIN}" - <<'PY'
import sys
sys.path.insert(0, "src")
from depth_anything_3.utils.constants import DTU_SCENES

if len(DTU_SCENES) != 22:
    raise SystemExit(f"Expected 22 DTU scenes, found {len(DTU_SCENES)}")
print(*DTU_SCENES, sep="\n")
PY
)

has_valid_metrics() {
    local scene="$1" result_root="$2"
    "${PYTHON_BIN}" - "${scene}" "${result_root}" <<'PY'
import json
import math
import sys
from pathlib import Path

scene, root = sys.argv[1:]
path = Path(root) / "metric_results/dtu_recon_unposed.json"
try:
    values = json.loads(path.read_text()).get(scene)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if not isinstance(values, dict) or not all(math.isfinite(values.get(k, float("nan"))) for k in ("acc", "comp", "overall")):
    raise SystemExit(1)
PY
}

verify_and_cleanup() {
    local scene="$1" result_root="$2" checkpoint_dir="$3"
    "${PYTHON_BIN}" - "${scene}" "${result_root}" <<'PY'
import json
import math
import sys
from pathlib import Path

scene, root = sys.argv[1:]
path = Path(root) / "metric_results/dtu_recon_unposed.json"
values = json.loads(path.read_text()).get(scene)
if not isinstance(values, dict) or not all(math.isfinite(values.get(k, float("nan"))) for k in ("acc", "comp", "overall")):
    raise SystemExit(f"Invalid reconstruction metrics for {scene}: {path}")
print(f"Verified {scene}: acc={values['acc']:.6f}, comp={values['comp']:.6f}, overall={values['overall']:.6f}")
PY

    local raw_dir="${result_root}/model_results"
    if [[ -d "${raw_dir}" ]]; then
        local bytes
        bytes="$(du -sb "${raw_dir}" | awk '{print $1}')"
        find "${raw_dir}" -depth -delete
        echo "Removed verified raw model results for ${scene}: ${bytes} bytes"
    fi

    # Retain only the final LoRA adapter and its camera-token sidecar.
    find "${checkpoint_dir}" -depth -mindepth 1 -maxdepth 1 \
        ! -name "epoch_$((TRAIN_EPOCHS - 1))_lora.pt" \
        ! -name "epoch_$((TRAIN_EPOCHS - 1))_lora_peft" -exec rm -rf -- {} +
}

for scene in "${SCENES[@]}"; do
    final_epoch=$((TRAIN_EPOCHS - 1))
    checkpoint_dir="${CHECKPOINT_ROOT}/${scene}"
    result_root="${WORKSPACE_ROOT}/${scene}/free_geometry_epoch${final_epoch}_seed${EVAL_SEED}"
    lora_path="${checkpoint_dir}/epoch_${final_epoch}_lora.pt"

    if has_valid_metrics "${scene}" "${result_root}"; then
        echo "Skipping completed scene: ${scene}"
        verify_and_cleanup "${scene}" "${result_root}" "${checkpoint_dir}"
        continue
    fi

    echo "============================================================"
    echo "DTU-49 per-scene LoRA reconstruction: ${scene} (49 views; camera token trainable)"
    echo "============================================================"
    "${PYTHON_BIN}" -u scripts/train_da3.py \
        --dataset dtu --scenes "${scene}" --samples_per_scene "${TRAIN_SAMPLES_PER_SCENE}" \
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
        --datasets dtu --modes recon_unposed --scenes "${scene}" --max_frames 49 --seed "${EVAL_SEED}" \
        --work_dir "${result_root}"
    verify_and_cleanup "${scene}" "${result_root}" "${checkpoint_dir}"
done

"${PYTHON_BIN}" - "${WORKSPACE_ROOT}" "${BASELINE_METRICS}" <<'PY'
import json
import sys
from pathlib import Path

root, baseline_path = map(Path, sys.argv[1:])
summary = {}
for path in sorted(root.glob("*/free_geometry_epoch*_seed*/metric_results/dtu_recon_unposed.json")):
    data = json.loads(path.read_text())
    summary.update({scene: values for scene, values in data.items() if scene != "mean"})
if len(summary) != 22:
    raise SystemExit(f"Expected 22 completed scene metrics, found {len(summary)}")
summary["mean"] = {key: sum(values[key] for scene, values in summary.items()) / len(summary) for key in ("acc", "comp", "overall")}
out = root / "metric_results" / "dtu_recon_unposed_per_scene_lora.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + "\n")

baseline = json.loads(baseline_path.read_text())
comparison = {
    "protocol": "full 49 views, seed 43, recon_unposed",
    "baseline_mean": baseline["mean"],
    "per_scene_lora_mean": summary["mean"],
    "delta_per_scene_lora_minus_baseline": {key: summary["mean"][key] - baseline["mean"][key] for key in ("acc", "comp", "overall")},
    "scenes": {scene: {"baseline": baseline[scene], "per_scene_lora": summary[scene], "delta_overall": summary[scene]["overall"] - baseline[scene]["overall"]} for scene in sorted(summary) if scene != "mean"},
}
comparison_path = root / "metric_results" / "dtu_recon_unposed_per_scene_comparison.json"
comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
print(f"Saved per-scene LoRA metrics: {out}")
print(f"Saved baseline comparison: {comparison_path}")
print(f"Mean overall: baseline={baseline['mean']['overall']:.6f}, per-scene LoRA={summary['mean']['overall']:.6f}")
PY
