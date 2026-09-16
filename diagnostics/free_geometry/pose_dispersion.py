#!/usr/bin/env python3
"""Pose-dispersion pre-probe for consensus rel-pose targets.

Three ideation agents converged on: average the teacher's RELATIVE poses
(gauge-invariant) over K resampled extra-view contexts. This probe measures,
per pair (GT offline only):
  - err_k: teacher rel-pose error vs GT for each of K contexts (Kabsch-aligned)
  - err_cons: consensus (chordal-mean rotation + mean direction) error vs GT
  - sigma: cross-context dispersion of the 6 rel-rotations (degrees)
  - gap: teacher-student rel-pose difference (= what distillation transfers)
Go if err_cons/err_single <= 0.80 or sigma/gap >= 0.2; kill if sigma/gap < 0.1.

Output: artifacts/diagnostics/pose_dispersion/summary.md
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, get_scene_data, load_manifest, frames_with_gt_depth, stable_seed  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import _kabsch_rt  # noqa: E402
import random

OUT = "artifacts/diagnostics/pose_dispersion"
K = 8
SCENES = ["1ada7a0617", "38d58a7a31", "7831862f02"]
PAIRS_PER_SCENE = 4
HW = (378, 504)


def rel_pairs(R, t):
    """6 unordered rel pairs: chordal-ready R_ij [6,3,3], t̂_ij [6,3]."""
    Rs, ts = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            Rs.append(R[i].T @ R[j])
            v = R[i].T @ (t[j] - t[i])
            ts.append(v / v.norm().clamp_min(1e-8))
    return torch.stack(Rs), torch.stack(ts)


def proj_so3(M_):
    U, _, Vh = torch.linalg.svd(M_)
    d = torch.sign(torch.linalg.det(U @ Vh))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    return U @ D @ Vh


def pair_err(Rs, ts, Rg, tg):
    rot = ((Rs - Rg) ** 2).sum(dim=(-2, -1)).sqrt().mean().item()
    ang = (1 - (ts * tg).sum(-1).clamp(-1, 1)).mean().item()
    return rot, ang


@torch.no_grad()
def main():
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    rows = []
    for scene in SCENES:
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        ok, _ = frames_with_gt_depth(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"][:PAIRS_PER_SCENE]):
            shared = list(pair["student_frames"])
            pool = [f for f in ok if f not in set(shared)]
            rng = random.Random(stable_seed(54_000, scene, pi))
            gt_ext = torch.from_numpy(np.asarray(scene_data.extrinsics)[shared]).float().cuda()
            Rg, tg = rel_pairs(gt_ext[:, :3, :3], gt_ext[:, :3, 3])
            images4 = torch.from_numpy(np.stack(
                [M.load_image_model(scene_data.image_files[i]) for i in shared], 0)
            ).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
            _, _, sp = M.student_preds(student, images4)
            es, _ = pose_encoding_to_extri_intri(sp["pose_enc"].float(), HW)
            Rs_s, ts_s = rel_pairs(es[0, :, :3, :3], es[0, :, :3, 3])
            Rs_all, ts_all = [], []
            for k in range(K):
                frames = [None] * 8
                for si, s in zip(STUDENT_INDICES, shared):
                    frames[si] = s
                for ei, e in zip([1, 3, 5, 7], rng.sample(pool, 4)):
                    frames[ei] = e
                imgs = torch.from_numpy(np.stack(
                    [M.load_image_model(scene_data.image_files[i]) for i in frames], 0)
                ).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
                feats24, psi = M.aggregator_all(teacher, imgs)
                p8 = M.replay_camera_nograd(teacher, feats24)
                et, _ = pose_encoding_to_extri_intri(p8.float(), HW)
                et = et[0, STUDENT_INDICES]
                Ra, ta = _kabsch_rt(et[:, :3, 3], gt_ext[:, :3, 3])
                Rs_k, ts_k = rel_pairs(Ra @ et[:, :3, :3], (Ra @ et[:, :3, 3].T).T + ta)
                Rs_all.append(Rs_k); ts_all.append(ts_k)
                del feats24
            # student 也对齐到 GT 再比（与 teacher 同口径）
            Ra_s, ta_s = _kabsch_rt(es[0, :, :3, 3], gt_ext[:, :3, 3])
            Rs_s, ts_s = rel_pairs(Ra_s @ es[0, :, :3, :3], (Ra_s @ es[0, :, :3, 3].T).T + ta_s)
            Rs_all = torch.stack(Rs_all); ts_all = torch.stack(ts_all)  # [K,6,3,3],[K,6,3]
            err_k = [pair_err(Rs_all[k], ts_all[k], Rg, tg) for k in range(K)]
            err_single = float(np.mean([e[0] + e[1] for e in err_k]))
            R_cons = torch.stack([proj_so3(Rs_all[:, p].mean(0)) for p in range(6)])
            t_cons = torch.nn.functional.normalize(ts_all.mean(0), dim=-1)
            err_cons = sum(pair_err(R_cons, t_cons, Rg, tg))
            err_stu = sum(pair_err(Rs_s, ts_s, Rg, tg))
            # dispersion: chordal std over K, deg-equivalent scale
            disp = (Rs_all - R_cons[None]).pow(2).sum(dim=(-2, -1)).sqrt().mean().item()
            rows.append(dict(scene=scene, pair=pi, single=err_single, cons=err_cons,
                             stu=err_stu, disp=disp))
            print(f"[{scene} p{pi}] single {err_single:.4f} cons {err_cons:.4f} "
                  f"ratio {err_cons/err_single:.3f} disp {disp:.4f} stu {err_stu:.4f}", flush=True)
        torch.cuda.empty_cache()
    import statistics
    r = [x["cons"] / x["single"] for x in rows]
    g = [x["disp"] / max(x["stu"] - x["single"], 1e-6) if x["stu"] > x["single"] else float("nan") for x in rows]
    lines = ["# pose dispersion pre-probe (K=8, 3 scenes x 4 pairs)", "",
             f"err(cons)/err(single): mean {statistics.mean(r):.3f} median {statistics.median(r):.3f}",
             f"dispersion/(student-teacher gap): mean {np.nanmean(g):.3f}",
             f"err_single {np.mean([x['single'] for x in rows]):.4f} err_cons {np.mean([x['cons'] for x in rows]):.4f} err_student {np.mean([x['stu'] for x in rows]):.4f}",
             "", "go if ratio<=0.80 or sigma/gap>=0.2; kill if sigma/gap<0.1"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
