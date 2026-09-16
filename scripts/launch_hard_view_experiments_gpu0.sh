#!/usr/bin/env bash
set -euo pipefail

# Texture is already materialized; GPU0 computes occlusion manifests, then
# serially trains/evaluates a distinct LoRA for each dataset x hard criterion.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0
# The cgroup grants 44 vCPUs. Keep 16 threads for this queue so Open3D/BLAS
# fusion accelerates without starving the independent GPU1 workload.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-$REPO_ROOT/model_weights/VGGT-1B}"
ARTIFACT_ROOT="$REPO_ROOT/artifacts/hard_view_subsets"
CHECKPOINT_ROOT="$REPO_ROOT/checkpoints/hard_view_subsets"
RESULT_ROOT="$REPO_ROOT/results/hard_view_subsets"

if ! "$PYTHON_BIN" - "$ARTIFACT_ROOT/eth3d/texture/eval.jsonl" "$ARTIFACT_ROOT/scannetpp/texture/eval.jsonl" <<'PY'
import sys
raise SystemExit(0 if all(sum(1 for _ in open(path)) == 25 for path in sys.argv[1:]) else 1)
PY
then
  "$PYTHON_BIN" scripts/build_hard_view_subsets.py --datasets eth3d scannetpp --criteria texture \
    --output-root "$ARTIFACT_ROOT" --candidate-count 128 --eval-samples-per-scene 5 --device cuda:0 --seed 30
fi

if ! "$PYTHON_BIN" - "$ARTIFACT_ROOT/eth3d/occlusion/selection_report.json" "$ARTIFACT_ROOT/scannetpp/occlusion/selection_report.json" <<'PY'
import json, sys
raise SystemExit(0 if all(json.load(open(path)).get("candidate_definition") for path in sys.argv[1:]) else 1)
PY
then
  "$PYTHON_BIN" scripts/build_hard_view_subsets.py --datasets eth3d scannetpp --criteria occlusion \
    --output-root "$ARTIFACT_ROOT" --candidate-count 128 --eval-samples-per-scene 5 --device cuda:0 --seed 30
else
  echo "[skip] completed occlusion manifests"
fi

common_train=(
  --samples_per_scene 10 --epochs 3 --batch_size 1 --num_workers 2 --image_size 504
  --model_name "$MODEL_NAME" --output_layers 4 11 17 23 --lora_rank 32 --lora_alpha 32
  --lora_layers_start 0 --lr 1e-5 --lr_scheduler cosine --warmup_ratio 0.15 --eta_min 1e-8
  --weight_decay 1e-5 --patch_huber_weight 1.0 --patch_huber_cos_weight 2.0
  --cf_weight 2.0 --cf_topk 4 --cf_num_ref_samples 256 --cf_num_shared_samples 256
  --cf_selection_mode mixed --use_cf_distance --cf_distance_weight 1.0
  --cf_distance_chunk_size 16 --cf_distance_type l2 --cf_distance_temperature 1.0
  --cf_distance_mode kl --cf_distance_huber_beta 0.5 --cf_d1_weight 1.0 --cf_d2_weight 1.0
  --cf_d3_weight 0.0 --save_interval 0 --log_interval 1 --seed 30
)

is_complete_train() {
  local run_dir="$1"
  [[ -s "$run_dir/latest_lora.pt" && -s "$run_dir/telemetry.json" ]] || return 1
  "$PYTHON_BIN" - "$run_dir/telemetry.json" <<'PY'
import json, sys
raise SystemExit(0 if len(json.load(open(sys.argv[1])).get("epochs", [])) == 3 else 1)
PY
}

for criterion in texture occlusion; do
  for dataset in eth3d scannetpp; do
    manifest="$ARTIFACT_ROOT/$dataset/$criterion/train.jsonl"
    run_dir="$CHECKPOINT_ROOT/$dataset/$criterion"
    if ! is_complete_train "$run_dir"; then
      "$PYTHON_BIN" scripts/train_vggt_covisibility_010.py --sequence-manifest "$manifest" --dataset "$dataset" \
        "${common_train[@]}" --output_dir "$run_dir"
    else
      echo "[skip] completed train: $dataset/$criterion"
    fi
    "$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py \
      --selection "$ARTIFACT_ROOT/$dataset/$criterion/source_frames.jsonl" --sequence-manifest "$ARTIFACT_ROOT/$dataset/$criterion/eval.jsonl" --datasets "$dataset" --model-name "$MODEL_NAME" \
      --lora-eth3d "$run_dir/latest_lora.pt" --lora-scannetpp "$run_dir/latest_lora.pt" \
      --output-root "$RESULT_ROOT/$dataset/$criterion" --recon-unposed --delete-arm-results
    "$PYTHON_BIN" scripts/summarize_hard_view_ci.py --window-metrics "$RESULT_ROOT/$dataset/$criterion/window_metrics.json" --dataset "$dataset"
    "$PYTHON_BIN" scripts/prune_hard_view_results.py \
      --result-root "$RESULT_ROOT/$dataset/$criterion" --dataset "$dataset"
  done
done

# This is intentionally a separate final evaluation: 16V is defined only for
# normalized occlusion and uses the already-trained matching occlusion LoRA.
for dataset in eth3d scannetpp; do
  subset="$ARTIFACT_ROOT/$dataset/occlusion"
  run_dir="$CHECKPOINT_ROOT/$dataset/occlusion"
  if [[ ! -s "$subset/eval_16v.jsonl" ]]; then
    "$PYTHON_BIN" scripts/build_normalized_occlusion_16v_eval.py --dataset "$dataset" \
      --artifact-root "$ARTIFACT_ROOT" --candidate-count 256 --samples-per-scene 5 --device cuda:0 --seed 130
  fi
  "$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py \
    --selection "$subset/source_frames.jsonl" --sequence-manifest "$subset/eval_16v.jsonl" --datasets "$dataset" --model-name "$MODEL_NAME" \
    --lora-eth3d "$run_dir/latest_lora.pt" --lora-scannetpp "$run_dir/latest_lora.pt" \
    --output-root "$RESULT_ROOT/$dataset/occlusion_16v" --recon-unposed --arms lora_16v --delete-arm-results
  "$PYTHON_BIN" scripts/prune_hard_view_results.py \
    --result-root "$RESULT_ROOT/$dataset/occlusion_16v" --dataset "$dataset" --required-arms lora_16v
done

echo "[done] $(date -u +%FT%TZ)"
