#!/usr/bin/env bash
set -euo pipefail

# Reconstruct the already-evaluated strict sequences to append F1/CD.  The
# training checkpoints and the exact audited manifests are reused unchanged.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"
MODEL_NAME="${MODEL_NAME:-$REPO_ROOT/model_weights/VGGT-1B}"
SELECTION="$REPO_ROOT/artifacts/covisibility_010_strict_top5/selected_scenes.jsonl"
MANIFEST_ROOT="$REPO_ROOT/artifacts/covisibility_010_strict_top5/sequences"
CHECKPOINT_ROOT="$REPO_ROOT/checkpoints/covisibility_010_strict_top5_vggt"
RESULT_ROOT="$REPO_ROOT/results/covisibility_010_strict_top5_vggt"

"$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py \
  --sequence-manifest "$MANIFEST_ROOT/eth3d_eval.jsonl" --selection "$SELECTION" \
  --datasets eth3d --model-name "$MODEL_NAME" --lora-eth3d "$CHECKPOINT_ROOT/eth3d/latest_lora.pt" \
  --output-root "$RESULT_ROOT/eth3d" --recon-unposed --delete-arm-results
"$PYTHON_BIN" scripts/benchmark_vggt_covisibility_010.py \
  --sequence-manifest "$MANIFEST_ROOT/scannetpp_eval.jsonl" --selection "$SELECTION" \
  --datasets scannetpp --model-name "$MODEL_NAME" --lora-scannetpp "$CHECKPOINT_ROOT/scannetpp/latest_lora.pt" \
  --output-root "$RESULT_ROOT/scannetpp" --recon-unposed --delete-arm-results
