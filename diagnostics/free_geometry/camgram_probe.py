#!/usr/bin/env python3
"""CamGram pre-probe: does the camera-token Gram distance (L23, after
camera_head.token_norm) predict the per-pair teacher-student pose AUC gap?
Spearman > 0.5 -> build arm; < 0.3 -> kill (agent-14 P3 gate).

Output: artifacts/diagnostics/camgram_probe/summary.md
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402

OUT = "artifacts/diagnostics/camgram_probe"


def gram_dist(cs, ct):
    """cs/ct [1,4,C] -> Frobenius dist of off-diagonal 4x4 cosine Gram."""
    a = torch.nn.functional.normalize(cs[0], dim=-1)
    b = torch.nn.functional.normalize(ct[0], dim=-1)
    Ga, Gb = a @ a.T, b @ b.T
    mask = ~torch.eye(4, dtype=torch.bool, device=a.device)
    return float(((Ga - Gb)[mask] ** 2).sum().sqrt())


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    base = M.get_base_vggt(student)
    rows = []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["probe_pairs"]):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            cache = M.cache_teacher_pair(teacher, images8, (ph, pw))
            feats24_s, psi, preds = M.student_preds(student, images4)
            ct = base.camera_head.token_norm(
                cache["feats"][23][:, STUDENT_INDICES, 0, :].float())
            cs = base.camera_head.token_norm(feats24_s[23][:, :, 0, :].float())
            gd = gram_dist(cs, ct)
            gt_ext = np.asarray(scene_data.extrinsics)[pair["student_frames"]]
            auc_t = M.pose_auc(cache["pose_enc8"][:, STUDENT_INDICES].float(), gt_ext)
            auc_s = M.pose_auc(preds["pose_enc"].float(), gt_ext)
            gap = float(auc_t["auc03"] - auc_s["auc03"])
            rows.append(dict(scene=scene, pair=pi, gram=gd, gap=gap,
                             auc_t=float(auc_t["auc03"]), auc_s=float(auc_s["auc03"])))
            print(f"[{scene} p{pi}] gram {gd:.4f} gap {gap:+.4f}", flush=True)
        torch.cuda.empty_cache()
    from scipy.stats import spearmanr
    g = np.array([r["gram"] for r in rows]); y = np.array([r["gap"] for r in rows])
    rho, p = spearmanr(g, y)
    lines = ["# camgram pre-probe (6 scenes x 2 probe pairs)", "",
             f"Spearman(gram_dist, teacher-student AUC@3 gap) = {rho:.3f} (p={p:.4f})",
             "", "| scene | pair | gram | gap |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['pair']} | {r['gram']:.4f} | {r['gap']:+.4f} |")
    lines += ["", "gate: >0.5 build arm, <0.3 kill"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:4]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
