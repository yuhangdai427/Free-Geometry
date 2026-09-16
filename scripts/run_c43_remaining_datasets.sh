#!/usr/bin/env bash
set -euo pipefail

# Run the three non-HiRoom experiments on GPU selected by the caller.
#
# Sequence: baseline evaluation -> formal Free-Geometry training/evaluation ->
# Test3R prompt TTA/evaluation.  Free-Geometry matches the maintained
# per-dataset training configurations; Test3R matches the HiRoom runner.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/workspace/benchmark_dataset}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace/c43_remaining_datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/c43_remaining_datasets/free_geometry_formal}"
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/model_weights/VGGT-1B}"
IMAGE_SIZE="${IMAGE_SIZE:-504}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS=(scannetpp 7scenes eth3d)
EVAL_SEED=43
EVAL_VIEWS="${EVAL_VIEWS:-4 8}"
TEST3R_MAX_TRIPLETS="${TEST3R_MAX_TRIPLETS:-100}"
TEST3R_TRIPLET_BATCH_SIZE="${TEST3R_TRIPLET_BATCH_SIZE:-4}"
COMMAND="${1:-all}"

require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "Missing required directory: ${path}" >&2
        return 1
    fi
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        return 1
    fi
}

require_nonempty_dir() {
    local path="$1"
    require_dir "${path}" || return 1
    if ! find "${path}" -mindepth 1 -type f -print -quit | grep -q .; then
        echo "No files found under: ${path}" >&2
        return 1
    fi
}

check_datasets() {
    local failed=0
    local scene
    local scannet_scenes=(
        09c1414f1b 1ada7a0617 40aec5fffa 3e8bba0176 acd95847c5
        578511c8a9 5f99900f09 7bc286c1b6 c4c04e6d6c c5439f4607
        286b55a2bf fb5a96b1a2 7831862f02 38d58a7a31 bde1e479ad
        9071e139d9 21d970d8de bcd2436daf cc5237fd77
    )
    local seven_scenes=(chess fire heads office pumpkin redkitchen stairs)
    local eth3d_scenes=(courtyard electro kicker pipes relief delivery_area facade office playground relief_2 terrains)

    echo "Dataset root: ${DATA_ROOT}"
    require_dir "${DATA_ROOT}" || failed=1

    for scene in "${scannet_scenes[@]}"; do
        require_nonempty_dir "${DATA_ROOT}/scannetpp/${scene}/merge_dslr_iphone/images" || failed=1
        require_dir "${DATA_ROOT}/scannetpp/${scene}/merge_dslr_iphone/colmap/sparse_render_rgb" || failed=1
        require_file "${DATA_ROOT}/scannetpp/${scene}/scans/mesh_aligned_0.05.ply" || failed=1
    done
    for scene in "${seven_scenes[@]}"; do
        require_nonempty_dir "${DATA_ROOT}/7scenes/7Scenes/${scene}" || failed=1
        require_file "${DATA_ROOT}/7scenes/7Scenes/meshes/${scene}.ply" || failed=1
    done
    for scene in "${eth3d_scenes[@]}"; do
        require_nonempty_dir "${DATA_ROOT}/eth3d/${scene}/images" || failed=1
        require_file "${DATA_ROOT}/eth3d/${scene}/dslr_calibration_jpg/cameras.txt" || failed=1
        require_file "${DATA_ROOT}/eth3d/${scene}/dslr_calibration_jpg/images.txt" || failed=1
        require_file "${DATA_ROOT}/eth3d/${scene}/combined_mesh.ply" || failed=1
    done

    if (( failed )); then
        echo "Dataset preflight failed. Finish the downloads at the paths above, then retry." >&2
        return 1
    fi
    echo "Dataset preflight passed for: ${DATASETS[*]}"
}

verify_matched_sampling() {
    local dataset="$1"
    local max_frames="$2"
    "${PYTHON_BIN}" - "${dataset}" "${max_frames}" "${EVAL_SEED}" <<'PY'
import random
import sys

# Both Test3R and benchmark_vggt call VGGTEvaluator._sample_frames(). This
# reproduces its non-subset branch and asserts that the common seed yields the
# same indices before either model is evaluated.
dataset, max_frames, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
examples = {
    "scannetpp": (397, "09c1414f1b"),
    "7scenes": (1000, "chess"),
    "eth3d": (38, "courtyard"),
}
num_frames, scene = examples[dataset]
if max_frames <= 0 or num_frames <= max_frames:
    indices = list(range(num_frames))
else:
    indices = list(range(num_frames))
    random.seed(seed)
    random.shuffle(indices)
    indices = sorted(indices[:max_frames])
print(f"Matched sampling preflight: {dataset}/{scene}, seed={seed}, {num_frames} -> {len(indices)} frames")
PY
}

verify_metrics() {
    local result_root="$1"
    "${PYTHON_BIN}" - "${result_root}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.rglob("*_pose.json")) + sorted(root.rglob("*_recon_unposed.json"))
if not files:
    raise SystemExit(f"No evaluation metric files under {root}")

def values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from values(value)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)

for path in files:
    numeric = list(values(json.loads(path.read_text())))
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise SystemExit(f"Invalid metrics: {path}")
    if not any(abs(value) > 0.0 for value in numeric):
        raise SystemExit(f"All-zero metrics: {path}")
print(f"Verified {len(files)} nonzero, finite metric files under {root}")
PY
}

cleanup_evaluation_outputs() {
    local result_root="$1"
    local before after

    # Metric JSON is the retained record after validation. Per-scene matrices,
    # point clouds, and visualizations are expensive intermediate outputs. The
    # evaluator nests these under seed43_*v, so remove every such subtree.
    before="$(du -sb "${result_root}" | awk '{print $1}')"
    while IFS= read -r -d '' output_dir; do
        rm -rf "${output_dir}"
    done < <(find "${result_root}" -type d \( -name model_results -o -name visualizations \) -print0)
    after="$(du -sb "${result_root}" | awk '{print $1}')"
    echo "Cleaned model_results and visualizations under ${result_root}: ${before} -> ${after} bytes"
}

get_free_geo_training_config() {
    case "$1" in
        scannetpp) echo "10 3 40 49 3e-5" ;;
        7scenes)   echo "10 3 40 49 1e-5" ;;
        eth3d)     echo "10 3 40 44 1e-5" ;;
        *) echo "Unknown Free-Geometry dataset: $1" >&2; return 1 ;;
    esac
}

run_test3r() {
    local dataset view_count result_root
    # All-frame inference over ScanNet++/7Scenes can exceed GPU0's memory
    # (hundreds to thousands of frames). Cap only those datasets at 100;
    # ETH3D keeps 0, the runner's all-frames setting.
    for dataset in scannetpp 7scenes eth3d; do
        if [[ "${dataset}" == "eth3d" ]]; then
            view_count=0
            result_root="${WORKSPACE_ROOT}/test3r_capped100/${dataset}/all_frames"
        else
            view_count=100
            result_root="${WORKSPACE_ROOT}/test3r_capped100/${dataset}/max100_frames"
        fi
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_vggt_test3r.py" \
            --base_model "${MODEL_NAME}" \
            --datasets "${dataset}" \
            --modes pose recon_unposed \
            --seeds "${EVAL_SEED}" \
            --view_counts "${view_count}" \
            --max_triplets "${TEST3R_MAX_TRIPLETS}" \
            --triplet_batch_size "${TEST3R_TRIPLET_BATCH_SIZE}" \
            --image_size "${IMAGE_SIZE}" \
            --work_dir "${result_root}"
        verify_metrics "${result_root}"
        cleanup_evaluation_outputs "${result_root}"
    done
}

run_baseline() {
    local n_views
    for n_views in ${EVAL_VIEWS}; do
        local max_frames="${n_views}"
        local eval_frames=()
        if [[ "${n_views}" == "4" ]]; then
            # Match the maintained 4v protocol: sample 8, then retain 4 even-indexed views.
            max_frames=8
            eval_frames=(--eval_frames 4)
        fi
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmark_vggt.py" \
            --base_model "${MODEL_NAME}" \
            --datasets "${DATASETS[@]}" \
            --modes pose recon_unposed \
            --seeds "${EVAL_SEED}" \
            --max_frames "${max_frames}" \
            --image_size "${IMAGE_SIZE}" \
            --work_dir "${WORKSPACE_ROOT}/baseline/seed${EVAL_SEED}_${n_views}v" \
            "${eval_frames[@]}"
        verify_metrics "${WORKSPACE_ROOT}/baseline/seed${EVAL_SEED}_${n_views}v"
        cleanup_evaluation_outputs "${WORKSPACE_ROOT}/baseline/seed${EVAL_SEED}_${n_views}v"
    done
}

run_100_frame_evaluations() {
    local dataset view_count result_root
    # Match the completed Test3R data protocol exactly: 100-frame caps for
    # ScanNet++/7Scenes and all frames for ETH3D (its scenes are small).
    for dataset in scannetpp 7scenes eth3d; do
        if [[ "${dataset}" == "eth3d" ]]; then
            view_count=0
            result_root="${WORKSPACE_ROOT}/baseline_freegeo_c43_matched/${dataset}/all_frames"
        else
            view_count=100
            result_root="${WORKSPACE_ROOT}/baseline_freegeo_c43_matched/${dataset}/max100_frames"
        fi
        verify_matched_sampling "${dataset}" "${view_count}"

        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmark_vggt.py" \
            --base_model "${MODEL_NAME}" --datasets "${dataset}" \
            --modes pose recon_unposed --seeds "${EVAL_SEED}" \
            --max_frames "${view_count}" --image_size "${IMAGE_SIZE}" \
            --work_dir "${result_root}/baseline"
        verify_metrics "${result_root}/baseline"
        cleanup_evaluation_outputs "${result_root}/baseline"

        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmark_vggt.py" \
            --lora_path "${CHECKPOINT_ROOT}/${dataset}/epoch_2_lora.pt" \
            --base_model "${MODEL_NAME}" --lora_rank 32 --lora_alpha 32 --lora_layers_start 0 \
            --datasets "${dataset}" --modes pose recon_unposed --seeds "${EVAL_SEED}" \
            --max_frames "${view_count}" --image_size "${IMAGE_SIZE}" \
            --work_dir "${result_root}/free_geometry"
        verify_metrics "${result_root}/free_geometry"
        cleanup_evaluation_outputs "${result_root}/free_geometry"
    done
}

run_free_geo() {
    local dataset n_views max_frames config samples epochs seed_start seed_end lr seeds final_epoch
    for dataset in "${DATASETS[@]}"; do
        config="$(get_free_geo_training_config "${dataset}")"
        read -r samples epochs seed_start seed_end lr <<< "${config}"
        seeds="$(seq -s ' ' "${seed_start}" "${seed_end}")"
        final_epoch=$((epochs - 1))
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_vggt.py" \
            --dataset "${dataset}" \
            --samples_per_scene "${samples}" --seeds_list ${seeds} \
            --model_name "${MODEL_NAME}" \
            --num_views 8 \
            --output_layers 4 11 17 23 \
            --patch_huber_weight 1.0 --patch_huber_cos_weight 2.0 --patch_huber_delta 1.0 \
            --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256 \
            --cf_angle1_weight 1.0 --cf_angle2_weight 1.0 --cf_angle3_weight 1.0 \
            --cf_selection_mode mixed --use_cf_distance \
            --cf_distance_weight 1.0 --cf_distance_chunk_size 16 --cf_distance_type l2 \
            --cf_distance_temperature 1.0 --cf_distance_mode kl --cf_distance_huber_beta 0.5 \
            --cf_d1_weight 1.0 --cf_d2_weight 1.0 --cf_d3_weight 0.0 \
            --epochs "${epochs}" --batch_size 2 --num_workers 2 --lr "${lr}" \
            --lora_rank 32 --lora_alpha 32 --lora_layers_start 0 \
            --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-8 --weight_decay 1e-5 \
            --output_dir "${CHECKPOINT_ROOT}/${dataset}" --log_interval 1 --save_interval 0

        for n_views in ${EVAL_VIEWS}; do
            max_frames="${n_views}"
            local eval_frames=()
            if [[ "${n_views}" == "4" ]]; then
                max_frames=8
                eval_frames=(--eval_frames 4)
            fi
            "${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmark_vggt.py" \
                --lora_path "${CHECKPOINT_ROOT}/${dataset}/epoch_${final_epoch}_lora.pt" \
                --base_model "${MODEL_NAME}" --lora_rank 32 --lora_alpha 32 --lora_layers_start 0 \
                --datasets "${dataset}" --modes pose recon_unposed --seeds "${EVAL_SEED}" \
                --max_frames "${max_frames}" --image_size "${IMAGE_SIZE}" \
                --work_dir "${WORKSPACE_ROOT}/free_geometry/seed${EVAL_SEED}_${n_views}v/${dataset}" \
                "${eval_frames[@]}"
            verify_metrics "${WORKSPACE_ROOT}/free_geometry/seed${EVAL_SEED}_${n_views}v/${dataset}"
            cleanup_evaluation_outputs "${WORKSPACE_ROOT}/free_geometry/seed${EVAL_SEED}_${n_views}v/${dataset}"
        done
    done
}

case "${COMMAND}" in
    check)
        check_datasets
        ;;
    baseline)
        check_datasets
        run_baseline
        ;;
    test3r)
        check_datasets
        run_test3r
        ;;
    free_geo)
        check_datasets
        run_free_geo
        ;;
    eval100)
        check_datasets
        run_100_frame_evaluations
        ;;
    all)
        check_datasets
        run_baseline
        run_free_geo
        run_test3r
        ;;
    remaining)
        check_datasets
        # Baseline completed before the interrupted Free-Geometry run. Validate
        # and reclaim its raw exports, then continue with the unfinished stages.
        for result_root in "${WORKSPACE_ROOT}"/baseline/seed${EVAL_SEED}_*v; do
            [[ -d "${result_root}" ]] || continue
            verify_metrics "${result_root}"
            cleanup_evaluation_outputs "${result_root}"
        done
        run_free_geo
        run_test3r
        ;;
    cleanup)
        for result_root in "${WORKSPACE_ROOT}"/baseline/seed${EVAL_SEED}_*v \
            "${WORKSPACE_ROOT}"/free_geometry/seed${EVAL_SEED}_*v/* \
            "${WORKSPACE_ROOT}"/test3r/* \
            "${WORKSPACE_ROOT}"/test3r_capped100/*/* \
            "${WORKSPACE_ROOT}"/baseline_freegeo_c43_matched/*/*; do
            [[ -d "${result_root}" ]] || continue
            verify_metrics "${result_root}"
            cleanup_evaluation_outputs "${result_root}"
        done
        ;;
    *)
        echo "Usage: $0 {check|baseline|test3r|free_geo|eval100|all|remaining|cleanup}" >&2
        exit 2
        ;;
esac
