#!/usr/bin/env bash
# Cross-product sweep: max_triplets x accum_iter over all HiRoom scenes.
#
# Grid:
#   max_triplets : 100, 200, 300
#   accum_iter   : 1, 2, 3
# = 9 runs. Each run TTA-adapts + evaluates 30 HiRoom scenes (seed 43,
# all frames, triplet_batch_size 4).
#
# Usage:
#   bash scripts/run_vggt_test3r_hiroom_sweep.sh
# (Self-detaches under nohup; returns immediately with PID + log path.)
#
# Env overrides:
#   MAX_TRIPLETS_GRID  - e.g. "100 200 300"
#   ACCUM_ITER_GRID    - e.g. "1 2 3"
#   WORK_ROOT          - parent dir for per-run WORK_DIRs
#                        (default workspace/vggt_test3r_hiroom_sweep)
#   LOG_DIR            - log directory (default ./logs)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/workspace/vggt_test3r_hiroom_sweep}"
MAX_TRIPLETS_GRID="${MAX_TRIPLETS_GRID:-100 200 300}"
ACCUM_ITER_GRID="${ACCUM_ITER_GRID:-1 2 3}"

mkdir -p "${LOG_DIR}" "${WORK_ROOT}"
LOGFILE="${LOG_DIR}/vggt_test3r_hiroom_sweep_${STAMP}.log"
PIDFILE="${LOG_DIR}/vggt_test3r_hiroom_sweep.pid"
SUMMARY="${WORK_ROOT}/sweep_summary_${STAMP}.tsv"

# ----- self-detach via nohup on first invocation -----------------------------
if [[ "${__DETACHED__:-0}" != "1" ]]; then
  echo "Launching detached sweep."
  echo "  log    : ${LOGFILE}"
  echo "  pid    : ${PIDFILE}"
  echo "  root   : ${WORK_ROOT}"
  echo "  grid   : max_triplets=[${MAX_TRIPLETS_GRID}] x accum_iter=[${ACCUM_ITER_GRID}]"
  __DETACHED__=1 nohup bash "${BASH_SOURCE[0]}" "$@" > "${LOGFILE}" 2>&1 &
  CHILD_PID=$!
  echo "${CHILD_PID}" > "${PIDFILE}"
  disown "${CHILD_PID}" 2>/dev/null || true
  echo "Started. PID=${CHILD_PID}"
  echo "Follow with: tail -f ${LOGFILE}"
  exit 0
fi

# ----- detached child: sweep -------------------------------------------------
echo "=================================================================="
echo "[$(date)] HiRoom sweep start"
echo "  grid: max_triplets=[${MAX_TRIPLETS_GRID}] x accum_iter=[${ACCUM_ITER_GRID}]"
echo "  work_root: ${WORK_ROOT}"
echo "=================================================================="

printf 'max_triplets\taccum_iter\twork_dir\tstatus\tauc03\tauc05\tauc15\tauc30\tfscore\toverall\n' > "${SUMMARY}"

for T in ${MAX_TRIPLETS_GRID}; do
  for A in ${ACCUM_ITER_GRID}; do
    TAG="T${T}_A${A}"
    RUN_WORK_DIR="${WORK_ROOT}/${TAG}"
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$(date)] Run ${TAG}  (max_triplets=${T}, accum_iter=${A})"
    echo "  work_dir: ${RUN_WORK_DIR}"
    echo "------------------------------------------------------------------"

    STATUS="ok"
    WORK_DIR="${RUN_WORK_DIR}" \
      bash "${REPO_ROOT}/scripts/run_vggt_test3r_hiroom.sh" \
        --max_triplets "${T}" \
        --accum_iter "${A}" \
      || STATUS="failed($?)"

    # Pull HiRoom means from the standard metric files.
    AUC03="NA"; AUC05="NA"; AUC15="NA"; AUC30="NA"; FSCORE="NA"; OVERALL="NA"
    POSE_JSON="${RUN_WORK_DIR}/seed43_0v/metric_results/hiroom_pose.json"
    RECON_JSON="${RUN_WORK_DIR}/seed43_0v/metric_results/hiroom_recon_unposed.json"
    if [[ -f "${POSE_JSON}" && -f "${RECON_JSON}" ]]; then
      read -r AUC03 AUC05 AUC15 AUC30 FSCORE OVERALL < <(python - "${POSE_JSON}" "${RECON_JSON}" <<'PY'
import json, sys
with open(sys.argv[1]) as f: pose = json.load(f).get("mean", {})
with open(sys.argv[2]) as f: recon = json.load(f).get("mean", {})
def g(d, *k):
    for key in k:
        if key in d: return d[key]
    return None
print(g(pose, "auc03", "Auc03", "auc3", "Auc3"),
      g(pose, "auc05", "Auc05"),
      g(pose, "auc15", "Auc15"),
      g(pose, "auc30", "Auc30"),
      g(recon, "fscore", "F1", "Fscore"),
      g(recon, "overall", "Overall"))
PY
)
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${T}" "${A}" "${RUN_WORK_DIR}" "${STATUS}" \
      "${AUC03}" "${AUC05}" "${AUC15}" "${AUC30}" "${FSCORE}" "${OVERALL}" \
      >> "${SUMMARY}"
  done
done

echo ""
echo "[$(date)] Sweep complete."
echo "  summary TSV: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
