#!/usr/bin/env bash
# Launch VGGT + Test3R-style prompt TTA on every HiRoom scene (all 30 scenes
# from selected_scene_list_val.txt) and evaluate in-place.
#
# The run uses whatever defaults live in scripts/run_vggt_test3r_hiroom.sh
# (currently seed 43, view_counts 0 = all frames, max_triplets 100,
# triplet_batch_size 4, modes pose + recon_unposed). This launcher just
# self-detaches via nohup, pipes logs to ./logs, and writes a PID file.
#
# Usage:
#   bash scripts/run_vggt_test3r_hiroom_all.sh
# (The script re-execs itself under nohup; no need to prepend `nohup ... &`.)
#
# Overrides via env vars:
#   WORK_DIR        - output directory (default workspace/vggt_test3r_hiroom_all)
#   LOG_DIR         - log directory (default ./logs)
#   MODEL_NAME      - HF model id (default facebook/vggt-1b)
#   IMAGE_SIZE      - input image size (default 504)
#
# Extra flags after `--` are forwarded to run_vggt_test3r.py, e.g.:
#   bash scripts/run_vggt_test3r_hiroom_all.sh -- --max_triplets 200 --accum_iter 1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/workspace/vggt_test3r_hiroom_all}"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"
LOGFILE="${LOG_DIR}/vggt_test3r_hiroom_all_${STAMP}.log"
PIDFILE="${LOG_DIR}/vggt_test3r_hiroom_all.pid"

# ----- self-detach under nohup on first invocation ---------------------------
if [[ "${__DETACHED__:-0}" != "1" ]]; then
  echo "Launching detached run."
  echo "  log : ${LOGFILE}"
  echo "  pid : ${PIDFILE}"
  echo "  out : ${WORK_DIR}"
  __DETACHED__=1 nohup bash "${BASH_SOURCE[0]}" "$@" > "${LOGFILE}" 2>&1 &
  CHILD_PID=$!
  echo "${CHILD_PID}" > "${PIDFILE}"
  disown "${CHILD_PID}" 2>/dev/null || true
  echo "Started. PID=${CHILD_PID}"
  echo "Follow with: tail -f ${LOGFILE}"
  exit 0
fi

# ----- detached child: run the full-scene sweep ------------------------------
echo "=================================================================="
echo "[$(date)] VGGT + Test3R TTA on HiRoom — all scenes"
echo "  repo      : ${REPO_ROOT}"
echo "  work_dir  : ${WORK_DIR}"
echo "  log       : ${LOGFILE}"
echo "  forwarded : $*"
echo "=================================================================="

# The underlying shell script (run_vggt_test3r_hiroom.sh) already iterates
# every scene listed in workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt
# because no --scenes filter is passed. Forward any extra CLI args.
WORK_DIR="${WORK_DIR}" \
MODEL_NAME="${MODEL_NAME:-facebook/vggt-1b}" \
IMAGE_SIZE="${IMAGE_SIZE:-504}" \
bash "${REPO_ROOT}/scripts/run_vggt_test3r_hiroom.sh" "$@"

STATUS=$?

echo ""
echo "[$(date)] Finished with exit code ${STATUS}."
echo "  summary : ${WORK_DIR}/vggt_test3r_summary.json"
echo "  prompts : ${WORK_DIR}/seed43_0v/prompts/"
echo "  metrics : ${WORK_DIR}/seed43_0v/metric_results/"

exit ${STATUS}
