#!/usr/bin/env python3
"""Geometric reprojection-verification probe (GT-free signal).

Hypothesis (post sim_probe): a shared-view patch is trustworthy where the
teacher's geometry is CROSS-VIEW VERIFIED — its depth backprojects from shared
view i into extra view j and lands on a consistent depth there. Unlike feature
similarity (rejected, AUC~0.52), this is a physical quantity and needs NO GT.

Signal per shared pixel: max over extra views j of [reprojection lands
in-bounds with |z - d_j|/z < 5%]; pooled per patch (mean).
Label (GT, offline): teacher 8v locally better than student 4v (pp_t < pp_s).
Report: AUC of geo score vs conf_t (0.5418) / sim (0.5187) baselines,
plus verified-patch coverage.

Output: artifacts/diagnostics/geo_probe/summary.md
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, TAP_LAYERS, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import perpatch_logres2, teacher_patch_conf  # noqa: E402

EXTRA_INDICES = [i for i in range(8) if i not in STUDENT_INDICES]
OUT = "artifacts/diagnostics/geo_probe"
STRIDE = 6
TOL = 0.05


def auc(scores, labels):
    m = np.isfinite(scores)
    s, y = scores[m], labels[m].astype(float)
    if len(y) == 0 or y.sum() in (0, len(y)):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def geo_scores(vggt, images8, patch_hw):
    """Per shared-view-patch geometric verification score [4,P] and coverage."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    feats24, psi = M.aggregator_all(vggt, images8)
    slots8 = [None] * 24
    for l in TAP_LAYERS:
        slots8[l] = feats24[l].float().contiguous()
    depth8, conf8 = M.replay_depth_nograd(vggt, slots8, images8, psi)
    pose8 = M.replay_camera_nograd(vggt, feats24)
    H, W = images8.shape[-2:]
    ext, intr = pose_encoding_to_extri_intri(pose8.float(), (H, W))
    del feats24
    B = 1
    d8 = depth8.squeeze(0).squeeze(-1).float()          # [8,H,W]
    R = ext[0, :, :3, :3].float(); t = ext[0, :, :3, 3].float()
    K = intr[0, :, :3, :3].float()                      # [8,3,3]
    ph, pw = patch_hw
    ys = torch.arange(STRIDE // 2, H, STRIDE, device=d8.device)
    xs = torch.arange(STRIDE // 2, W, STRIDE, device=d8.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(gx)
    pix = torch.stack([gx, gy, ones], -1).float()       # [h,w,3]
    scores = torch.zeros(4, len(ys), len(xs), device=d8.device)
    for vi, si in enumerate(STUDENT_INDICES):
        d_i = d8[si][ys][:, xs]                          # [h,w]
        Xc = (torch.linalg.inv(K[si]) @ pix.reshape(-1, 3, 1)).squeeze(-1) * d_i.reshape(-1, 1)
        Xw = (R[si] @ Xc.T).T + t[si]                    # [N,3]
        best = torch.zeros(Xw.shape[0], device=d8.device)
        for ej in EXTRA_INDICES:
            Xj = (R[ej].T @ (Xw - t[ej]).T).T            # [N,3]
            z = Xj[:, 2].clamp_min(1e-6)
            uv = (K[ej] @ Xj.T).T
            uv = uv[:, :2] / z[:, None]
            inb = (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1) & (Xj[:, 2] > 0)
            grid = torch.stack([2 * uv[:, 0] / (W - 1) - 1, 2 * uv[:, 1] / (H - 1) - 1], -1)
            dj = F.grid_sample(d8[ej][None, None], grid[None, None, :, :],
                               align_corners=True, padding_mode="zeros")[0, 0, 0, 0]
            cons = inb & ((z - dj).abs() / z < TOL)
            best = torch.maximum(best, cons.float())
        scores[vi] = best.reshape(len(ys), len(xs))
    P = ph * pw
    pid = (ys[:, None] // (H // ph)) * pw + (xs[None, :] // (W // pw))
    pid = pid.reshape(-1)
    flat = scores.reshape(4, -1)
    sc = torch.zeros(4, P, device=d8.device)
    cnt = torch.zeros(P, device=d8.device)
    sc.index_add_(1, pid, flat)
    cnt.index_add_(0, pid, torch.ones_like(pid, dtype=d8.dtype))
    sc = sc / cnt.clamp_min(1)[None]
    return sc, depth8, conf8


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    G, Y, C, SC_, PAIR = [], [], [], [], []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"] + sc["probe_pairs"]):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            gsc, depth8, conf8 = geo_scores(teacher, images8, (ph, pw))
            _, _, spreds = M.student_preds(student, images4)
            sdepth = spreds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"], images8.shape[-2:])
            d4 = depth8.squeeze(0).squeeze(-1).float().cpu().numpy()[STUDENT_INDICES]
            pp_t = perpatch_logres2(d4, gt4, (ph, pw))
            pp_s = perpatch_logres2(sdepth, gt4, (ph, pw))
            conf_t = teacher_patch_conf({"conf4": conf8[:, STUDENT_INDICES]}, (ph, pw)).squeeze(0).cpu().numpy()
            G.append(gsc.cpu().numpy().reshape(-1))
            Y.append((pp_t < pp_s).reshape(-1).astype(float))
            C.append(conf_t.reshape(-1))
            SC_.append(scene); PAIR += [pi] * 0
            print(f"[{scene} p{pi}] geo cover={float(gsc.mean()):.3f}", flush=True)
        torch.cuda.empty_cache()
    G = np.concatenate(G); Y = np.concatenate(Y); C = np.concatenate(C)
    lines = ["# geo verification probe", "",
             f"pairs: 6 scenes x 12, patch coverage(mean geo score) = {G.mean():.3f}", "",
             "| signal | AUC(pooled) |", "|---|---|",
             f"| geo_reproj (GT-free) | {auc(G, Y):.4f} |",
             f"| teacher conf | {auc(C, Y):.4f} |",
             f"| geo x conf | {auc(G * C, Y):.4f} |",
             "| sim_max (from sim_probe) | 0.5187 |",
             "| base rate | %.3f |" % float(Y.mean())]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
