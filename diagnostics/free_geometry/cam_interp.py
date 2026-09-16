#!/usr/bin/env python3
"""Camera-token alpha-interpolation probe (D1-style, for the camera path).

User challenge: "swap the student's camera tokens for the teacher's and the
result must improve". Endpoint data exists (7/12 better, +0.39 max, -0.22 min).
This probe traces the PATH: replace token-0 of every aggregator layer with
h_a = (1-a)*student + a*teacher on the shared 4 views, replay the frozen
camera head, measure pose AUC@3 vs GT for a in {0,.25,.5,.75,1}, plus SHAM
(a-blend with teacher tokens from a different pair; must degrade).

Output: artifacts/diagnostics/cam_interp/summary.md
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402

OUT = "artifacts/diagnostics/cam_interp"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    rows = []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student); student.eval()
        pairs = sc["train_pairs"] + sc["probe_pairs"]
        for pi, pair in enumerate(pairs):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            tfeats, _ = M.aggregator_all(teacher, images8)
            sfeats, _ = student._get_aggregator()(images4)
            sfeats = [f.float() for f in sfeats]
            pair_sham = pairs[(pi + 3) % len(pairs)]
            images8_sh, _ = M.load_pair_images(scene_data, pair_sham["teacher_frames"])
            tfeats_sh, _ = M.aggregator_all(teacher, images8_sh)
            gt_ext = np.asarray(scene_data.extrinsics)[pair["student_frames"]]
            base = M.get_base_vggt(student)

            def auc_with(t_src, alpha):
                blended = []
                for l in range(24):
                    if t_src[l] is None:
                        blended.append(None)
                        continue
                    f = sfeats[l].float().clone()
                    tt = t_src[l][:, STUDENT_INDICES, 0, :].float()
                    f[:, :, 0, :] = (1 - alpha) * f[:, :, 0, :] + alpha * tt
                    blended.append(f)
                pose = M.replay_camera_nograd(base, blended)
                return M.pose_auc(pose.float(), gt_ext)["auc03"]

            for sham, tsrc in ((0, tfeats), (1, tfeats_sh)):
                for a in ALPHAS:
                    auc = auc_with(tsrc, a)
                    rows.append(dict(scene=scene, pair=pi, sham=sham, alpha=a, auc=auc))
                print(f"[{scene} p{pi} sham{sham}] " +
                      " ".join(f"a{a}:{auc_with(tsrc, a):.3f}" for a in ALPHAS), flush=True)
            del tfeats, sfeats, tfeats_sh
            torch.cuda.empty_cache()
    import collections, statistics
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["sham"], r["alpha"])].append(r["auc"])
    lines = ["# camera-token alpha interpolation (6 scenes x 12 pairs)", "",
             "| alpha | real AUC@3 | sham AUC@3 |", "|---|---|---|"]
    for a in ALPHAS:
        lines.append(f"| {a} | {statistics.mean(agg[(0, a)]):.4f} | {statistics.mean(agg[(1, a)]):.4f} |")
    real_better = sum(1 for r0, r1 in zip(
        [r for r in rows if r["sham"] == 0 and r["alpha"] == 0.0],
        [r for r in rows if r["sham"] == 0 and r["alpha"] == 1.0]) if r1["auc"] > r0["auc"])
    lines += ["", f"endpoint (a=1) better than a=0: {real_better}/12 pairs",
              "path valid if real AUC rises with alpha and sham degrades"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
