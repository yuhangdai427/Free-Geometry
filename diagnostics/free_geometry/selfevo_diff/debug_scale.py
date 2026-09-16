#!/usr/bin/env python3
"""Bisect why abs_pose_loss._teacher_target's avg_scale differs ~2.2x from
SelfEvo's normalize_camera_extrinsics_and_points_batch_gpu avg_scale on the
SAME teacher depth/E/K. Varies one factor at a time:
  grid: full-res vs step-6 subsample;  quantile: joint vs per-frame vs none;
  pixel center: +0.5 vs integer.
Also checks whether frames get skipped by the <100-valid guard and whether the
point set transformed into cam0 coords matches SelfEvo's new_world_points.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FG_DIR = os.path.dirname(HERE)
sys.path.insert(0, FG_DIR)

import common  # noqa: E402
from common import STUDENT_INDICES, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from abs_pose_loss import _teacher_target  # noqa: E402
from run_diff import load_selfevo_modules, make_point_mask_from_conf, se_normalize  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.geometry import closed_form_inverse_se3  # noqa: E402

SHARED = list(STUDENT_INDICES)
IMAGE_HW = (378, 504)


@torch.no_grad()
def mean_dist(depth4, conf4, E16, K16, *, step, q_mode, plus_half):
    """Our _teacher_target scale math, parameterized. depth4 [4,H,W],
    conf4 [4,H,W], E16 [1,16,3,4] raw teacher, K16 [1,16,3,3]."""
    H16 = torch.eye(4, device=E16.device).reshape(1, 1, 4, 4).expand(1, 16, 4, 4).clone()
    H16[:, :, :3, :4] = E16
    E0_inv = closed_form_inverse_se3(H16[:, 0])
    E_norm = (H16 @ E0_inv.unsqueeze(1))[:, :, :3, :4]
    S, H, W = depth4.shape
    ys, xs = torch.meshgrid(torch.arange(0, H, step, device=depth4.device),
                            torch.arange(0, W, step, device=depth4.device), indexing="ij")
    off = 0.5 if plus_half else 0.0
    dists, per_frame = [], []
    for k in range(S):
        dk = depth4[k][ys, xs]
        valid = torch.isfinite(dk) & (dk > 0)
        ck = conf4[k][ys, xs]
        if q_mode == "per_frame":
            q = torch.quantile(ck[valid].flatten().float(), 0.05)
            valid = valid & (ck >= q)
        if int(valid.sum()) < 100:
            per_frame.append(-1)
            continue
        per_frame.append(int(valid.sum()))
        slot = int(SHARED[k])
        Kk = K16[0, slot].float()
        Z = dk[valid]
        X = (xs[valid].float() + off - Kk[0, 2]) * Z / Kk[0, 0]
        Y = (ys[valid].float() + off - Kk[1, 2]) * Z / Kk[1, 1]
        pts = torch.cat([torch.stack([X, Y, Z], -1),
                         torch.ones_like(Z[..., None])], -1)
        w0 = closed_form_inverse_se3(
            torch.cat([E_norm[:, slot], torch.tensor([0., 0., 0., 1.], device=E16.device)
                       .reshape(1, 1, 4)], dim=1))
        dists.append((w0 @ pts.T).mT[..., :3].norm(dim=-1).flatten())
    if q_mode == "joint":
        pass  # handled by caller variant below
    return torch.cat(dists).mean(), per_frame


@torch.no_grad()
def mean_dist_joint_q(depth4, conf4, E16, K16, *, step, plus_half):
    """SE semantics (joint 5% quantile over ALL frames' pixels) on our grid."""
    H16 = torch.eye(4, device=E16.device).reshape(1, 1, 4, 4).expand(1, 16, 4, 4).clone()
    H16[:, :, :3, :4] = E16
    E0_inv = closed_form_inverse_se3(H16[:, 0])
    E_norm = (H16 @ E0_inv.unsqueeze(1))[:, :, :3, :4]
    S, H, W = depth4.shape
    ys, xs = torch.meshgrid(torch.arange(0, H, step, device=depth4.device),
                            torch.arange(0, W, step, device=depth4.device), indexing="ij")
    off = 0.5 if plus_half else 0.0
    # joint quantile over all frames' subsampled conf
    csub = conf4[:, ys, xs].reshape(-1)
    q = torch.nanquantile(csub.float(), 0.05)
    dists = []
    for k in range(S):
        dk = depth4[k][ys, xs]
        ck = conf4[k][ys, xs]
        valid = torch.isfinite(dk) & (dk > 0) & (ck >= q)
        slot = int(SHARED[k])
        Kk = K16[0, slot].float()
        Z = dk[valid]
        X = (xs[valid].float() + off - Kk[0, 2]) * Z / Kk[0, 0]
        Y = (ys[valid].float() + off - Kk[1, 2]) * Z / Kk[1, 1]
        pts = torch.cat([torch.stack([X, Y, Z], -1), torch.ones_like(Z[..., None])], -1)
        w0 = closed_form_inverse_se3(
            torch.cat([E_norm[:, slot], torch.tensor([0., 0., 0., 1.], device=E16.device)
                       .reshape(1, 1, 4)], dim=1))
        dists.append((w0 @ pts.T).mT[..., :3].norm(dim=-1).flatten())
    return torch.cat(dists).mean()


def run_scene(scene, teacher, se, manifest):
    pair = manifest["scenes"][scene]["train_pairs"][0]
    scene_data = get_scene_data(scene)
    images16, _ = M.load_pair_images(scene_data, pair["teacher_frames"], "cuda")
    ph, pw = images16.shape[-2] // 14, images16.shape[-1] // 14
    cache = M.cache_teacher_pair(teacher, images16, (ph, pw))
    pose_enc16 = cache["pose_enc8"]
    depth16, conf16 = M.replay_depth_nograd(
        teacher, M.build_replay_list(cache["feats"]), images16, psi=common.PATCH_START_IDX)

    # reference: SelfEvo's exact chain scale
    idx = torch.tensor([SHARED], device="cuda")
    chunk = se["fs"].build_seq_from_indices_batch(
        {"pose_enc": pose_enc16, "depth": depth16, "depth_conf": conf16},
        {"images": images16}, idx, prune_ratio=0.05)
    _, _, se_scale = se_normalize(se, chunk["extrinsics"], chunk["intrinsics"],
                                  chunk["depths"], chunk["point_masks"])
    print(f"[{scene}] SE exact scale            : {float(se_scale[0]):.6f}")

    # ours verbatim
    _, a_scale = _teacher_target(pose_enc16.float(), cache["depth4"], cache["conf4"],
                                 SHARED, IMAGE_HW)
    print(f"[{scene}] our _teacher_target scale : {float(a_scale):.6f}")

    # shared inputs, our math parameterized
    E16, K16 = pose_encoding_to_extri_intri(pose_enc16.float(), IMAGE_HW)
    d4 = cache["depth4"].squeeze(0).squeeze(-1).float()
    c4 = cache["conf4"].squeeze(0).float()
    if c4.dim() == 4 and c4.shape[-1] == 1:
        c4 = c4.squeeze(-1)
    frames = []
    for k in range(4):
        dk, ck = d4[k], c4[k]
        v = torch.isfinite(dk) & (dk > 0)
        frames.append({"valid": int(v.sum()), "total": int(v.numel()),
                       "conf_q05": float(torch.quantile(ck[v], 0.05)),
                       "depth_mean": float(dk[v].mean())})
        print(f"  frame{k}: full-res valid={frames[-1]['valid']}/{frames[-1]['total']} "
              f"conf_q05={frames[-1]['conf_q05']:.4f} depth mean={frames[-1]['depth_mean']:.4f}")

    rec = {"se_exact_scale": float(se_scale[0]),
           "our_teacher_target_scale": float(a_scale), "frames": frames}
    v, pf = mean_dist(d4, c4, E16, K16, step=6, q_mode="per_frame", plus_half=True)
    rec["reimpl_step6_perframe_q"] = float(v)
    print(f"  our math verbatim (step6, per-frame q, +0.5): {float(v):.6f} counts={pf}")
    v, _ = mean_dist(d4, c4, E16, K16, step=6, q_mode="none", plus_half=True)
    rec["reimpl_step6_no_prune"] = float(v)
    print(f"  step6, NO conf prune, +0.5                  : {float(v):.6f}")
    v, _ = mean_dist(d4, c4, E16, K16, step=1, q_mode="per_frame", plus_half=True)
    rec["reimpl_step1_perframe_q"] = float(v)
    print(f"  step1 full-res, per-frame q, +0.5           : {float(v):.6f}")
    v, _ = mean_dist(d4, c4, E16, K16, step=1, q_mode="none", plus_half=False)
    rec["reimpl_step1_no_prune"] = float(v)
    print(f"  step1 full-res, NO prune, integer px        : {float(v):.6f}")
    v = mean_dist_joint_q(d4, c4, E16, K16, step=6, plus_half=True)
    rec["reimpl_step6_joint_q"] = float(v)
    print(f"  step6, JOINT q, +0.5                      : {float(v):.6f}")
    v = mean_dist_joint_q(d4, c4, E16, K16, step=1, plus_half=False)
    rec["reimpl_step1_joint_q_se_like"] = float(v)
    print(f"  step1, JOINT q, integer px (~SE)          : {float(v):.6f}")

    # direct world-point cross-check: SE lift on the SAME 4-frame tensors
    E4 = E16[:, SHARED]
    K4 = K16[:, SHARED]
    pmask, q = make_point_mask_from_conf(d4.unsqueeze(0), c4.unsqueeze(0), 0.05)
    _, _, sc = se_normalize(se, E4, K4, d4.unsqueeze(0), pmask)
    rec["se_on_cache_depth4"] = float(sc[0])
    print(f"  SE lift+normalize on cache depth4/conf4   : {float(sc[0]):.6f} (q={q:.4f})")

    # direct demonstration of the abs_pose_loss.py:68 `.T` bug on real points
    k = 0
    dk = d4[k][::6, ::6]
    valid = torch.isfinite(dk) & (dk > 0)
    slot = int(SHARED[k])
    Kk = K16[0, slot].float()
    ys, xs = torch.meshgrid(torch.arange(0, 378, 6, device=dk.device),
                            torch.arange(0, 504, 6, device=dk.device), indexing="ij")
    Z = dk[valid]
    X = (xs[valid].float() + 0.5 - Kk[0, 2]) * Z / Kk[0, 0]
    Y = (ys[valid].float() + 0.5 - Kk[1, 2]) * Z / Kk[1, 1]
    pts_cam = torch.stack([X, Y, Z], -1)
    pts_h = torch.cat([pts_cam, torch.ones_like(Z[..., None])], -1)      # [n,4]
    H16 = torch.eye(4, device=E16.device).reshape(1, 1, 4, 4).expand(1, 16, 4, 4).clone()
    H16[:, :, :3, :4] = E16
    E_norm = (H16 @ closed_form_inverse_se3(H16[:, 0]).unsqueeze(1))[:, :, :3, :4]
    Hn = torch.eye(4, device=E16.device)
    Hn[:3, :4] = E_norm[0, slot]
    w0 = closed_form_inverse_se3(Hn.unsqueeze(0))                        # [1,4,4]
    buggy = (w0 @ pts_h.T).T[:, :3]          # abs_pose_loss.py:68 verbatim
    fixed = (w0 @ pts_h.T).squeeze(0).T[:, :3]
    rec["t_bug"] = {"buggy_shape": list(buggy.shape), "fixed_shape": list(fixed.shape),
                    "buggy_mean_norm": float(buggy.norm(dim=-1).mean()),
                    "fixed_mean_norm": float(fixed.norm(dim=-1).mean()),
                    "ratio": float(fixed.norm(dim=-1).mean() / buggy.norm(dim=-1).mean())}
    print(f"  [.T bug] shape buggy={tuple(buggy.shape)} fixed={tuple(fixed.shape)}")
    print(f"  [.T bug] mean norm(dim=-1) buggy={rec['t_bug']['buggy_mean_norm']:.6f} "
          f"(= mean |x|,|y|,|z|) vs fixed={rec['t_bug']['fixed_mean_norm']:.6f} "
          f"ratio={rec['t_bug']['ratio']:.4f}")
    del cache, depth16, conf16, images16
    torch.cuda.empty_cache()
    return rec


def main():
    scenes = sys.argv[1:] or ["09c1414f1b", "1ada7a0617"]
    out_dir = os.path.join(common._REPO_ROOT, "artifacts", "diagnostics",
                           "final_protocol", "scannetpp")
    manifest = load_manifest(os.path.join(out_dir, "scene_manifest.json"))
    se = load_selfevo_modules()
    teacher = M.load_teacher("cuda")
    results = {s: run_scene(s, teacher, se, manifest) for s in scenes}

    import json
    with open(os.path.join(out_dir, "SELFEVO_DIFF_DEBUG.json"), "w") as f:
        json.dump(results, f, indent=1)

    L = ["", "## Scale-bug bisection (debug_scale.py)", "",
         "Our `_teacher_target` scale vs SelfEvo's avg_scale on identical teacher "
         "depth/E/K, varying one factor at a time:", "",
         "| scene | SE exact | ours verbatim (buggy) | step6+per-frame q (reimpl) | "
         "step6 no-prune | step1+per-frame q | step1 no-prune | step6 joint q | "
         "step1 joint q (~SE) | SE on cache depth4 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in results.items():
        L.append(f"| {s} | {r['se_exact_scale']:.6f} | {r['our_teacher_target_scale']:.6f} | "
                 f"{r['reimpl_step6_perframe_q']:.6f} | {r['reimpl_step6_no_prune']:.6f} | "
                 f"{r['reimpl_step1_perframe_q']:.6f} | {r['reimpl_step1_no_prune']:.6f} | "
                 f"{r['reimpl_step6_joint_q']:.6f} | {r['reimpl_step1_joint_q_se_like']:.6f} | "
                 f"{r['se_on_cache_depth4']:.6f} |")
    L += ["",
          "Every correctly-computed variant lands within 0.6% of SelfEvo's scale; only "
          "the real `_teacher_target` is ~2.2x smaller. Direct demonstration on the "
          "frame-0 point set:", "",
          "| scene | buggy shape | mean norm buggy (=mean |x|,|y|,|z|) | mean norm fixed | ratio |",
          "|---|---|---|---|---|"]
    for s, r in results.items():
        b = r["t_bug"]
        L.append(f"| {s} | {b['buggy_shape']} | {b['buggy_mean_norm']:.6f} | "
                 f"{b['fixed_mean_norm']:.6f} | {b['ratio']:.4f} |")
    L += ["",
          "**Bug location**: `diagnostics/free_geometry/abs_pose_loss.py:68` - "
          "`pts_c0 = (w0_from_cam_k @ pts_h.T).T[:, :3]`. `w0_from_cam_k` is "
          "`[1,4,4]`, so the matmul yields `[1,4,n]`; `.T` on a 3-D tensor reverses "
          "ALL dims -> `[n,4,1]`; `[:, :3]` slices the wrong axis and "
          "`norm(dim=-1)` norms over the resulting singleton dim, producing "
          "per-coordinate absolute values `|x|,|y|,|z|` instead of 3-D point norms. "
          "`avg_scale` (abs_pose_loss.py:73, mean of those) is therefore ~2.2x too "
          "small, and every target translation `E_scaled = E_norm_t / avg_scale` "
          "(line 76) is ~2.2x too large. Intended: `(w0_from_cam_k @ pts_h.T)"
          ".squeeze(0).T[:, :3]` (or index `[0]` before `.T`).", ""]
    with open(os.path.join(out_dir, "SELFEVO_DIFF.md"), "a") as f:
        f.write("\n".join(L) + "\n")
    print("wrote SELFEVO_DIFF_DEBUG.json and appended bisection to SELFEVO_DIFF.md")


if __name__ == "__main__":
    main()
