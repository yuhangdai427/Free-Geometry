#!/usr/bin/env bash
set -euo pipefail

# Resumable GPU0 launcher for the two Co-visibility-0.1 LoRA trainings, then
# their four-arm evaluations.  Logs stay in the repository for inspection.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-$REPO_ROOT/model_weights/VGGT-1B}"
MANIFEST_ROOT="$REPO_ROOT/artifacts/covisibility_010_sequences/full"
CHECKPOINT_ROOT="$REPO_ROOT/checkpoints/covisibility_010_vggt"
RESULT_ROOT="$REPO_ROOT/results/covisibility_010_vggt/full"

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

is_complete() {
  local run_dir="$1"
  [[ -s "$run_dir/latest_lora.pt" && -s "$run_dir/telemetry.json" ]] || return 1
  "$PYTHON_BIN" - "$run_dir/telemetry.json" <<'PY'
import json, sys
raise SystemExit(0 if len(json.load(open(sys.argv[1])).get("epochs", [])) == 3 else 1)
PY
}

train() {
  local dataset="$1"
  local run_dir="$CHECKPOINT_ROOT/$dataset"
  if is_complete "$run_dir"; then
    echo "[skip] completed training: $dataset"
    return
  fi
  mkdir -p "$run_dir"
  "$PYTHON_BIN" scripts/train_vggt_covisibility_010.py --sequence-manifest "$MANIFEST_ROOT/${dataset}_train.jsonl" --dataset "$dataset" "${common_train[@]}" --output_dir "$run_dir"
}

train eth3d
train scannetpp

"$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py \
  --sequence-manifest "$MANIFEST_ROOT/eth3d_eval.jsonl" --datasets eth3d --model-name "$MODEL_NAME" \
  --lora-eth3d "$CHECKPOINT_ROOT/eth3d/latest_lora.pt" --output-root "$RESULT_ROOT/eth3d" --recon-unposed
"$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py \
  --sequence-manifest "$MANIFEST_ROOT/scannetpp_eval.jsonl" --datasets scannetpp --model-name "$MODEL_NAME" \
  --lora-scannetpp "$CHECKPOINT_ROOT/scannetpp/latest_lora.pt" --output-root "$RESULT_ROOT/scannetpp" --recon-unposed

echo "[done] $(date -u +%FT%TZ)"
