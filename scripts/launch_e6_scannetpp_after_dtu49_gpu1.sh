#!/usr/bin/env bash
set -euo pipefail

# Wait for the user-owned DTU49 GPU1 tmux job, then run the strict P16/P32
# ScanNet++ extension of E6 in a separate session on GPU1.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SESSION="${WAIT_SESSION:-da3_dtu49_per_scene_lora_gpu1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/da3/bin/python}"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  echo "[$(date -u +%FT%TZ)] waiting for ${WAIT_SESSION} to finish"
  sleep 60
done

echo "[$(date -u +%FT%TZ)] ${WAIT_SESSION} finished; starting ScanNet++ E6 on GPU1"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=1
export DATASETS_OVERRIDE="scannetpp"
export TRAIN_LR="3e-5"
export OUTPUT_ROOT="${REPO_ROOT}/checkpoints/e6_nested_views_scannetpp"
export RESULTS_ROOT="${REPO_ROOT}/results/e6_nested_views_scannetpp"
export INDICES_ROOT="${REPO_ROOT}/artifacts/e6_nested_views_indices"
# This cgroup is capped at 44 CPUs and 220 GiB RAM.  Sixteen CPU-only TSDF
# fusion threads leave capacity for the evaluator, inference, and other jobs.
export FUSION_WORKERS="${FUSION_WORKERS:-16}"
export PYTHON_BIN
exec "${REPO_ROOT}/scripts/launch_e6_nested_views_resume.sh"
