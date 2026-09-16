#!/bin/bash
# Free-Geometry overnight experiment queue (2026-09-10).
# Every phase is isolated: a failure is logged and the queue continues.
set -uo pipefail
cd /root/autodl-tmp/Free-Geometry
D=diagnostics/free_geometry
A=artifacts/diagnostics
LOG=$A/overnight.log
mkdir -p $A

say()  { echo "[$(date '+%m-%d %H:%M')] $1" | tee -a $LOG; }
guard() { # guard <name> <cmd...>
  local name="$1"; shift
  local free_gb=$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt 30 ]; then say "ABORT $name: only ${free_gb}G disk left"; return 1; fi
  say "START $name"
  if "$@" >> $LOG 2>&1; then say "OK $name"; else say "FAILED $name (see log)"; fi
}

# ---------- Phase B: 20-scene confirmation (32v) ----------
guard "B manifest-20" python3 $D/build_manifest.py --run_root $A/bakeoff_20scenes --n_scenes 20

NEW_SCENES=$(python3 - <<'EOF'
import json
m20 = json.load(open("artifacts/diagnostics/bakeoff_20scenes/scene_manifest.json"))
m6 = json.load(open("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json"))
print(" ".join(sorted(set(m20["scenes"]) - set(m6["scenes"]))))
EOF
)
say "B new scenes: $NEW_SCENES"

# train 4 arms on the 14 NEW scenes (6 dev scenes reuse v2/v4/v5 results)
guard "B train (14 new scenes x 4 arms)" python3 $D/train_arms.py \
  --run_root $A/bakeoff_20scenes --scenes $NEW_SCENES \
  --arms A1_deployed B5_conf C2_b5_rel G4_conf_l23only

# link the 6 dev scenes' existing npz into the 20-scene tree (scene-level links)
python3 - <<'EOF'
import os
pairs = {  # source exp -> 20scenes exp name
  "bakeoff_v2_transductive": ["A0_baseline", "A1_deployed_step30", "A1_deployed_step100"],
  "bakeoff_v4_fixes": ["B5_conf_step30", "B5_conf_step100"],
  "bakeoff_v5_poserel": ["C2_b5_rel_step30", "C2_b5_rel_step100"],
  "bakeoff_v6_gates": ["G4_conf_l23only_step30", "G4_conf_l23only_step100"],
}
m6 = __import__("json").load(open("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json"))
for src, exps in pairs.items():
    for exp in exps:
        for scene in m6["scenes"]:
            src_dir = f"artifacts/diagnostics/{src}/eval32/{exp}/model_results/scannetpp/{scene}"
            dst_dir = f"artifacts/diagnostics/bakeoff_20scenes/eval32/{exp}/model_results/scannetpp/{scene}"
            if os.path.isdir(src_dir) and not os.path.exists(dst_dir):
                os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                os.symlink(os.path.abspath(src_dir), dst_dir)
print("linked dev-scene npz")
EOF

guard "B eval 32v (20 scenes)" python3 $D/run_eval.py --run_root $A/bakeoff_20scenes \
  --manifest $A/bakeoff_20scenes/scene_manifest.json
guard "B depth metrics" python3 $D/depth_metrics.py --run_root $A/bakeoff_20scenes \
  --manifest $A/bakeoff_20scenes/scene_manifest.json

# ---------- Phase B2: 8v/4v eval on all 20 scenes from ckpts ----------
guard "B2 lowv infer" python3 $D/eval_viewcounts.py \
  --run_root $A/bakeoff_20scenes_lowv \
  --manifest $A/bakeoff_20scenes/scene_manifest.json \
  --ckpt_root $A/bakeoff_20scenes/ckpts \
  --arms A1_deployed B5_conf C2_b5_rel G4_conf_l23only
cp $A/bakeoff_20scenes/scene_manifest.json $A/bakeoff_20scenes_lowv/
guard "B2 lowv eval" python3 $D/run_eval.py --run_root $A/bakeoff_20scenes_lowv

# ---------- Phase C: capacity & mechanism (6 dev scenes) ----------
guard "C3 full-finetune" python3 $D/train_arms.py --run_root $A/bakeoff_v7_mech \
  --scenes $(python3 diagnostics/free_geometry/_six_scenes.py) \
  --arms B5_conf --full_ft --no_ckpt
guard "C5 camtok unfrozen" python3 $D/train_arms.py --run_root $A/bakeoff_v7_mech \
  --scenes $(python3 diagnostics/free_geometry/_six_scenes.py) \
  --arms C2_b5_rel --train_camera_token --no_ckpt
guard "C4 selfmix EMA-probe" python3 $D/train_arms.py --run_root $A/bakeoff_v7_mech \
  --scenes $(python3 diagnostics/free_geometry/_six_scenes.py) \
  --arms C4_selfmix --no_ckpt
guard "C eval" python3 $D/run_eval.py --run_root $A/bakeoff_v7_mech \
  --manifest $A/bakeoff_v2_transductive/scene_manifest.json

# ---------- Phase D: new supervision forms (6 dev scenes) ----------
guard "D new forms" python3 $D/train_arms.py --run_root $A/bakeoff_v8_forms \
  --scenes $(python3 diagnostics/free_geometry/_six_scenes.py) \
  --arms D1_relfeat D2_layermean D3_spatialpool D4_depthgrad --no_ckpt
guard "D eval" python3 $D/run_eval.py --run_root $A/bakeoff_v8_forms \
  --manifest $A/bakeoff_v2_transductive/scene_manifest.json
guard "D depth metrics" python3 $D/depth_metrics.py --run_root $A/bakeoff_v8_forms \
  --manifest $A/bakeoff_v2_transductive/scene_manifest.json

# ---------- Phase E: disk cleanup (keep metrics only) ----------
say "F cleanup"
python3 - <<'EOF'
import os, shutil, glob
# delete model_results npz trees where metric jsons exist (metrics already extracted)
for tree in glob.glob("artifacts/diagnostics/bakeoff_*/eval32"):
    for exp in os.listdir(tree):
        mr = os.path.join(tree, exp, "metric_results", "scannetpp_pose.json")
        modelres = os.path.join(tree, exp, "model_results")
        if os.path.exists(mr) and os.path.isdir(modelres):
            shutil.rmtree(modelres, ignore_errors=True)
# delete losing-arm ckpts (keep B winners: B5/C2/G4 step30 in 20scenes tree)
for v in ["bakeoff_v1", "bakeoff_v3_featurespace", "bakeoff_v6_gates"]:
    shutil.rmtree(f"artifacts/diagnostics/{v}/ckpts", ignore_errors=True)
print("cleanup done")
EOF
say "=== overnight queue finished $(date) ==="
