#!/bin/bash
# Recover Phase-4 CamRel/CONFD metrics (their eval json was overwritten by the
# next arm's run_eval after npz cleanup). Re-infer npz from saved ckpts,
# re-evaluate just those arms, keep per-batch metric files.
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
DG=diagnostics/free_geometry
P4=artifacts/diagnostics/final_protocol_phase4
for ds in scannetpp 7scenes hiroom eth3d; do
  RR=$P4/$ds
  case "$ds" in scannetpp|7scenes) V=100v;; *) V=allv;; esac
  cp "$RR/eval32_metrics.json" "$RR/eval32_metrics_CTM.bak.json" 2>/dev/null || true
  echo "=== [$ds] re-infer CamRel/CONFD $(date '+%F %T') ===" >> "$RR/stream_recover.log"
  python "$DG/eval_viewcounts.py" --manifest "$RR/scene_manifest.json" \
      --ckpt_root "$RR/ckpts" --run_root "$RR" --step 100 \
      --arms C2M_CamRel CONFD_REL --view_subsets "$V" >> "$RR/stream_recover.log" 2>&1
  python "$DG/run_eval.py" --run_root "$RR" --datas "$ds" \
      --experiments "C2M_CamRel@$V" "CONFD_REL@$V" >> "$RR/stream_recover.log" 2>&1
  python "$DG/depth_metrics.py" --run_root "$RR" \
      --manifest "$RR/scene_manifest.json" --out "$RR/depth_metrics_camrelconfd.csv" \
      >> "$RR/stream_recover.log" 2>&1
  mv "$RR/eval32_metrics.json" "$RR/eval32_metrics_camrelconfd.json"
  cp "$RR/eval32_metrics_CTM.bak.json" "$RR/eval32_metrics.json" 2>/dev/null || true
  rm -rf "$RR"/eval32/*/model_results
  echo "=== [$ds] recover DONE $(date '+%F %T') ===" >> "$RR/stream_recover.log"
done
echo "RECOVER COMPLETE"
