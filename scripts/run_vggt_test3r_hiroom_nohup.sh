#!/usr/bin/env bash
# Launcher for VGGT+Test3R on HiRoom (all frames, seeds 43/44/45).
#
# Stage 1: Run the plain VGGT baseline (no TTA). If the reported
#          HiRoom pose Auc3 matches BASELINE_AUC3_TARGET (within
#          BASELINE_AUC3_TOL), continue; otherwise stop so the
#          user can investigate.
#
# Stage 2: Run VGGT + Test3R-style prompt TTA on the same scenes
#          and seeds. Since --view_counts 0 sends all frames into
#          both the baseline and the adapted path, the two runs
#          see identical inputs per scene/seed.
#
# Usage:
#   nohup bash scripts/run_vggt_test3r_hiroom_nohup.sh \
#       > logs/vggt_test3r_hiroom_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Or, the short form below already wraps the nohup/log redirection.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

BASELINE_WORK_DIR="${BASELINE_WORK_DIR:-${REPO_ROOT}/workspace/vggt_baseline_hiroom_allframes}"
ADAPTED_WORK_DIR="${ADAPTED_WORK_DIR:-${REPO_ROOT}/workspace/vggt_test3r_hiroom_allframes}"

# Expected baseline HiRoom pose Auc3 from a prior run (shared by both pipelines
# because view_counts=0 sends every frame in every scene to the model).
BASELINE_AUC3_TARGET="${BASELINE_AUC3_TARGET:-0.47777777777777786}"
BASELINE_AUC3_TOL="${BASELINE_AUC3_TOL:-1e-6}"

PIDFILE="${LOG_DIR}/vggt_test3r_hiroom.pid"
LOGFILE="${LOG_DIR}/vggt_test3r_hiroom_${STAMP}.log"

# ---- If not already detached, re-exec ourselves under nohup ----------------
if [[ "${__DETACHED__:-0}" != "1" ]]; then
  echo "Launching detached run. Log: ${LOGFILE}"
  __DETACHED__=1 nohup bash "${BASH_SOURCE[0]}" "$@" \
      > "${LOGFILE}" 2>&1 &
  CHILD_PID=$!
  echo "${CHILD_PID}" > "${PIDFILE}"
  disown "${CHILD_PID}" 2>/dev/null || true
  echo "PID: ${CHILD_PID} (saved to ${PIDFILE})"
  echo "tail -f ${LOGFILE}"
  exit 0
fi

echo "=================================================================="
echo "[$(date)] VGGT+Test3R HiRoom launcher"
echo "  baseline work_dir : ${BASELINE_WORK_DIR}"
echo "  adapted  work_dir : ${ADAPTED_WORK_DIR}"
echo "  seeds             : 43 44 45"
echo "  target Auc3       : ${BASELINE_AUC3_TARGET} (tol ${BASELINE_AUC3_TOL})"
echo "=================================================================="

BASELINE_METRIC_FILE="${BASELINE_WORK_DIR}/seed43_0v/metric_results/hiroom_pose.json"

run_baseline() {
  echo "[$(date)] >>> Stage 1: baseline (--no_ttt)"
  WORK_DIR="${BASELINE_WORK_DIR}" bash "${REPO_ROOT}/scripts/run_vggt_test3r_hiroom.sh" \
      --no_ttt
  echo "[$(date)] <<< baseline finished"
}

run_adapted() {
  echo "[$(date)] >>> Stage 2: VGGT + Test3R-style TTA"
  WORK_DIR="${ADAPTED_WORK_DIR}" bash "${REPO_ROOT}/scripts/run_vggt_test3r_hiroom.sh"
  echo "[$(date)] <<< adapted run finished"
}

check_auc3() {
  # Read HiRoom pose Auc3 (mean) from the seed43 baseline metrics and
  # compare against BASELINE_AUC3_TARGET with BASELINE_AUC3_TOL tolerance.
  if [[ ! -f "${BASELINE_METRIC_FILE}" ]]; then
    echo "[WARN] No baseline metrics at ${BASELINE_METRIC_FILE}; cannot auto-verify."
    return 2
  fi
  python - "${BASELINE_METRIC_FILE}" "${BASELINE_AUC3_TARGET}" "${BASELINE_AUC3_TOL}" <<'PY'
import json, sys
path, target, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with open(path) as f:
    data = json.load(f)
mean = data.get("mean", {})
auc3 = mean.get("Auc3") or mean.get("auc3") or mean.get("Auc@3")
if auc3 is None:
    print(f"[ERROR] No Auc3 key in {path}: keys={list(mean.keys())}")
    sys.exit(1)
diff = abs(auc3 - target)
status = "MATCH" if diff <= tol else "MISMATCH"
print(f"[baseline] HiRoom pose Auc3 mean = {auc3} (target {target}, diff {diff:.3e}) -> {status}")
sys.exit(0 if diff <= tol else 1)
PY
}

# Stage 1 -------------------------------------------------------------------
if [[ -f "${BASELINE_METRIC_FILE}" ]] && check_auc3; then
  echo "[$(date)] Baseline metrics already present and match target. Skipping stage 1."
else
  run_baseline
  if ! check_auc3; then
    echo "[$(date)] Baseline Auc3 does not match ${BASELINE_AUC3_TARGET}. Stopping before stage 2."
    echo "To force stage 2, re-run with BASELINE_AUC3_TOL=1 (or inspect ${BASELINE_METRIC_FILE})."
    exit 1
  fi
fi

# Stage 2 -------------------------------------------------------------------
run_adapted

echo "[$(date)] All done."
echo "  baseline summary: ${BASELINE_WORK_DIR}/vggt_test3r_summary.json"
echo "  adapted  summary: ${ADAPTED_WORK_DIR}/vggt_test3r_summary.json"
