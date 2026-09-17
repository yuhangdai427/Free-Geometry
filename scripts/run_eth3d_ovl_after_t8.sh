#!/bin/bash
# Orchestrator: wait for the VGGT t8 chain to finish, then run the queued
# DA3 eth3d sparse_overlap experiment (covisibility-constrained teacher windows),
# then compute paired-vs-baseline deltas into a result summary.
set -u
cd /root/autodl-tmp/Free-Geometry
PY=/root/miniconda3/envs/da3/bin/python
LOG=logs/vggt_t8_chain.log

echo "[orch] waiting for t8 chain $(date '+%F %T')"
while pgrep -f "run_vggt_t8_all.sh" > /dev/null; do sleep 60; done
if grep -q "ALL DONE" "$LOG"; then
  echo "[orch] chain completed OK $(date '+%F %T')"
else
  echo "[orch] WARNING: chain process gone WITHOUT 'ALL DONE' (crashed?) $(date '+%F %T')"
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[orch] launching eth3d sparse_overlap experiment $(date '+%F %T')"
"$PY" scripts/train_da3_protocol.py --dataset eth3d \
  --scenes courtyard delivery_area electro facade kicker office pipes playground relief relief_2 terrains \
  --output_root workspace/da3_protocol_eth3d_t8s4_ovl \
  --arm rkdc1h --teacher_N 8 --steps 100 --sparse_overlap \
  > logs/da3_eth3d_ovl.log 2>&1
rc=$?
echo "[orch] experiment exit=$rc $(date '+%F %T'); computing summary"
"$PY" - <<'EOF'
import json
import numpy as np
s = json.load(open('workspace/da3_protocol_eth3d_t8s4_ovl/smoke_summary.json'))['scenes']
ba = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/eth3d_baseline.json'))['scenes']
br = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/eth3d_recon_baseline.json'))['scenes']
da, df, lines = [], [], []
for sc, r in sorted(s.items()):
    b, f = ba[sc], br[sc]
    d1 = (r['eval']['auc03']-b['auc03'])/b['auc03']*100
    d2 = (r['eval']['recon_fscore']-f['fscore'])/f['fscore']*100
    da.append(d1); df.append(d2)
    lines.append(f"| {sc} | {d1:+.2f} | {d2:+.2f} |")
out = ["# DA3 eth3d 8:4 + sparse_overlap (covisibility teacher windows)", "",
       f"Mean dAUC@3 = **{np.mean(da):+.2f}%**  dF1 = **{np.mean(df):+.2f}%**  (n={len(da)})",
       "", "Reference: plain 8:4 champion = +2.95% / -1.21%", "",
       "| scene | dAUC% | dF1% |", "|---|---|---|", *lines]
open('workspace/da3_protocol_eth3d_t8s4_ovl/result_summary.md', 'w').write("\n".join(out))
print(f"SUMMARY dAUC={np.mean(da):+.2f}% dF1={np.mean(df):+.2f}% (plain 8:4: +2.95/-1.21)")
EOF
echo "[orch] ALL ORCHESTRATION DONE $(date '+%F %T')"
