#!/usr/bin/env python3
"""D1 (teacher-target interpolation + sham control + camera endpoints) and
D2 (equal-norm loss-direction test, global and per-layer variants, plus
output-space direction) on the frozen 12 probe pairs.

No model parameters are updated anywhere in this script.

Outputs (appended):
  <run_root>/feature_interpolation.csv
  <run_root>/feature_direction.csv
  <run_root>/camera_endpoints.csv
  <run_root>/patch_audit.csv
"""

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import (
    STUDENT_INDICES, TAP_LAYERS, e_depth, get_scene_data, load_manifest,
)
import modeling as M

ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]
BETAS = [0.01, 0.03, 0.1]
EPS = 1e-8


def insert_patch_feats(student_feats24, patch_by_layer: Dict[int, torch.Tensor]):
    """24-slot list: patched layers get special tokens from student + new patch
    tokens; other slots None. patch_by_layer values: [1,4,P,2048]."""
    slots = [None] * 24
    for layer, hp in patch_by_layer.items():
        special = student_feats24[layer][:, :, : M.PATCH_START_IDX, :]
        slots[layer] = torch.cat([special, hp], dim=2)
    return slots


def replay_edepth(vggt, slots, images4, gt4):
    depth, _ = M.replay_depth_nograd(vggt, slots, images4)
    d = depth.squeeze(0).squeeze(-1).float().cpu().numpy()
    e, n = e_depth(d, gt4)
    return e, n


def three_space_mses(depth_head, patch_by_layer, teacher_cache, patch_hw):
    raw, nrm, prj = 0.0, 0.0, 0.0
    with torch.no_grad():
        for layer in TAP_LAYERS:
            hs = patch_by_layer[layer]
            ht = M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float())
            raw += torch.mean((hs - ht) ** 2).item()
            nrm += torch.mean((M.to_norm(depth_head, hs) - M.to_norm(depth_head, ht)) ** 2).item()
            prj += torch.mean(
                (M.to_projected(depth_head, hs, layer, patch_hw)
                 - M.to_projected(depth_head, ht, layer, patch_hw)) ** 2
            ).item()
    n = len(TAP_LAYERS)
    return raw / n, nrm / n, prj / n


def d1_for_pair(vggt, scene_data, pair, cache, other_cache, images8, images4, gt4, patch_hw):
    rows = []
    base = vggt
    # student = frozen 4-view forward of the SAME frozen backbone
    feats24_s, psi = M.aggregator_all(vggt, images4)
    H0 = {l: M.to_patch(feats24_s[l].float()) for l in TAP_LAYERS}
    HT = {l: M.to_patch(cache["feats"][l][:, STUDENT_INDICES].float()) for l in TAP_LAYERS}
    HS = {l: M.to_patch(other_cache["feats"][l][:, STUDENT_INDICES].float()) for l in TAP_LAYERS}

    for tag, target in (("teacher", HT), ("sham", HS)):
        for alpha in ALPHAS:
            Ha = {l: H0[l] + alpha * (target[l] - H0[l]) for l in TAP_LAYERS}
            slots = insert_patch_feats(feats24_s, Ha)
            e, n = replay_edepth(vggt, slots, images4, gt4)
            raw, nrm, prj = three_space_mses(base.depth_head, Ha, cache, patch_hw)
            rows.append(dict(alpha=alpha, sham=int(tag == "sham"), e_depth=e, n_valid=n,
                             mse_raw=raw, mse_norm=nrm, mse_proj=prj))
    return rows, feats24_s, H0, HT


def camera_endpoints(vggt, scene_data, pair, cache, images4):
    from modeling import pose_auc

    gt_ext = np.asarray(scene_data.extrinsics)[pair["student_frames"]]
    feats24_s, psi = M.aggregator_all(vggt, images4)
    pose_s = M.replay_camera_nograd(vggt, feats24_s)          # 4-view camera head
    pose_t = cache["pose_enc8"][:, STUDENT_INDICES]            # teacher 8-view context
    auc_s = pose_auc(pose_s, gt_ext)
    auc_t = pose_auc(pose_t, gt_ext)
    return {f"student_{k}": v for k, v in auc_s.items()} | {f"teacher_{k}": v for k, v in auc_t.items()}


# ---------------------------------------------------------------- D2

def loss_values(depth_head, H, HT, patch_hw, teacher_depth4, vggt, images4, psi, student_feats24):
    """All four losses evaluated at patch-feature dict H (tensors, may require grad)."""
    out = {}
    for l in TAP_LAYERS:
        hs, ht = H[l], HT[l]
        out.setdefault("raw", 0.0)
        out["raw"] = out["raw"] + torch.mean((hs - ht) ** 2)
        ns, nt = M.to_norm(depth_head, hs), M.to_norm(depth_head, ht)
        out.setdefault("norm", 0.0)
        out["norm"] = out["norm"] + torch.mean((ns - nt) ** 2)
        zs = M.to_projected(depth_head, hs, l, patch_hw)
        zt = M.to_projected(depth_head, ht, l, patch_hw)
        out.setdefault("proj", 0.0)
        out["proj"] = out["proj"] + torch.mean((zs - zt) ** 2)
    n = len(TAP_LAYERS)
    for k in ("raw", "norm", "proj"):
        out[k] = out[k] / n

    # output-space: scale-invariant log-depth MSE vs teacher 8->4 depth
    slots = insert_patch_feats(student_feats24, H)
    with torch.autocast(device_type="cuda", enabled=False):
        depth, _ = vggt.depth_head(slots, images=images4, patch_start_idx=psi)
    depth = depth.squeeze(0).squeeze(-1)          # [4,H,W]
    tdepth = teacher_depth4.squeeze(0).squeeze(-1)
    valid = torch.isfinite(tdepth) & (tdepth > 0)
    r = torch.log(depth.clamp_min(1e-6)) - torch.log(tdepth.clamp_min(1e-6))
    r = torch.where(valid, r, torch.zeros_like(r))
    c = r.sum() / valid.sum().clamp_min(1)
    out["out"] = (((r - c) ** 2) * valid).sum() / valid.sum().clamp_min(1)
    return out


def d2_for_pair(vggt, scene_data, pair, cache, images8, images4, gt4, patch_hw, feats24_s, H0, HT):
    depth_head = vggt.depth_head
    rows = []
    psi = M.PATCH_START_IDX

    e0, _ = replay_edepth(vggt, insert_patch_feats(feats24_s, H0), images4, gt4)
    losses0 = {k: float(v) for k, v in loss_values(
        depth_head, {l: H0[l] for l in TAP_LAYERS}, HT, patch_hw,
        cache["depth4"], vggt, images4, psi, feats24_s).items()}

    # gradients at H0 (fresh leaf each loss)
    grads = {}
    for name in ("raw", "norm", "proj", "out"):
        H_leaf = {l: H0[l].clone().requires_grad_(True) for l in TAP_LAYERS}
        lv = loss_values(depth_head, H_leaf, HT, patch_hw, cache["depth4"], vggt, images4, psi, feats24_s)
        g = torch.autograd.grad(lv[name], [H_leaf[l] for l in TAP_LAYERS])
        grads[name] = {l: g[i].detach() for i, l in enumerate(TAP_LAYERS)}
        del H_leaf, lv, g
    torch.cuda.empty_cache()

    G_global = torch.sqrt(sum(((HT[l] - H0[l]) ** 2).sum() for l in TAP_LAYERS)).item()
    G_layer = {l: torch.sqrt(((HT[l] - H0[l]) ** 2).sum()).item() for l in TAP_LAYERS}

    for name in ("raw", "norm", "proj", "out"):
        g = grads[name]
        gnorm_global = torch.sqrt(sum((g[l] ** 2).sum() for l in TAP_LAYERS)).item()
        gnorm_layer = {l: torch.sqrt((g[l] ** 2).sum()).item() for l in TAP_LAYERS}
        # per-layer gradient RMS (for the layer-scale confound check)
        rms = {l: float(torch.sqrt((g[l] ** 2).mean()).item()) for l in TAP_LAYERS}

        for variant in ("global", "perlayer"):
            for beta in BETAS:
                if gnorm_global < EPS:
                    continue
                Hn = {}
                for l in TAP_LAYERS:
                    if variant == "global":
                        step = beta * G_global * g[l] / (gnorm_global + EPS)
                    else:
                        step = beta * G_layer[l] * g[l] / (gnorm_layer[l] + EPS)
                    Hn[l] = H0[l] - step
                e1, _ = replay_edepth(vggt, insert_patch_feats(feats24_s, Hn), images4, gt4)
                lv1 = {k: float(v) for k, v in loss_values(
                    depth_head, Hn, HT, patch_hw, cache["depth4"], vggt, images4, psi, feats24_s).items()}
                upd = torch.sqrt(sum(((Hn[l] - H0[l]) ** 2).sum() for l in TAP_LAYERS)).item()
                rows.append(dict(
                    loss=name, variant=variant, beta=beta,
                    e_depth_before=e0, e_depth_after=e1,
                    own_loss_before=losses0[name], own_loss_after=lv1[name],
                    raw_before=losses0["raw"], raw_after=lv1["raw"],
                    norm_before=losses0["norm"], norm_after=lv1["norm"],
                    proj_before=losses0["proj"], proj_after=lv1["proj"],
                    out_before=losses0["out"], out_after=lv1["out"],
                    update_norm=upd,
                    overshoot=int(lv1[name] > losses0[name]),
                    grad_rms_l4=rms[4], grad_rms_l11=rms[11],
                    grad_rms_l17=rms[17], grad_rms_l23=rms[23],
                ))
    return rows, e0, losses0, grads, G_global


def patch_audit(H0, HT, grads, e_perpatch_s, e_perpatch_t):
    """Top-10% patch shares of loss and gradient energy; teacher-good/bad split."""
    rows = []
    for l in TAP_LAYERS:
        # per-patch squared residual, mean over channels: [1,4,P]
        res = ((H0[l] - HT[l]) ** 2).mean(dim=-1)
        g = grads["raw"][l]
        ge = (g ** 2).sum(dim=-1)
        for scope, v in (("loss", res), ("grad", ge)):
            flat = v.reshape(-1)
            k = max(1, flat.numel() // 10)
            top_share = flat.topk(k).values.sum().item() / flat.sum().clamp_min(1e-12).item()
            rows.append(dict(layer=l, scope=scope, top10_share=top_share))
    # teacher local quality vs residual mass (pair-wide scale corrections)
    rs2 = e_perpatch_s  # [4,P] squared log-residual (centered), student
    rt2 = e_perpatch_t
    delta = rs2 - rt2  # >0: teacher better
    better = delta > 0
    res_l23 = ((H0[23] - HT[23]) ** 2).mean(dim=-1).squeeze(0).cpu().numpy()  # [4,P]
    tot = max(res_l23.sum(), 1e-12)
    rows.append(dict(layer=23, scope="loss_share_teacher_better",
                     top10_share=(res_l23[better].sum() / tot).item()))
    rows.append(dict(layer=23, scope="loss_share_teacher_worse",
                     top10_share=(res_l23[~better].sum() / tot).item()))
    rows.append(dict(layer=23, scope="frac_patches_teacher_better",
                     top10_share=float(better.mean())))
    return rows


def perpatch_logres2(depth, gt4, patch_hw):
    """[4,H,W] depth vs gt -> per-patch centered squared log residual [4,P]."""
    ph, pw = patch_hw
    r = np.log(np.clip(depth, 1e-6, None)) - np.log(np.clip(gt4, 1e-6, None))
    valid = np.isfinite(gt4) & (gt4 > 0)
    r = np.where(valid, r, 0.0)
    c = r.sum() / max(valid.sum(), 1)
    r2 = np.where(valid, (r - c) ** 2, 0.0)
    S, H, W = r2.shape
    return r2.reshape(S, ph, H // ph, pw, W // pw).mean(axis=(2, 4)).reshape(S, ph * pw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default="artifacts/diagnostics/bakeoff_v1")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    manifest = load_manifest(os.path.join(args.run_root, "scene_manifest.json"))
    scenes = args.scenes or sorted(manifest["scenes"])

    paths = {k: os.path.join(args.run_root, k)
             for k in ("feature_interpolation.csv", "feature_direction.csv",
                       "camera_endpoints.csv", "patch_audit.csv")}
    files = {k: open(p, "a", newline="") for k, p in paths.items()}
    writers: Dict[str, csv.DictWriter] = {}
    written_header = {k: os.path.exists(p) and os.path.getsize(p) > 0 for k, p in paths.items()}

    def write(kind, row):
        if kind not in writers:
            writers[kind] = csv.DictWriter(files[kind], fieldnames=list(row.keys()))
            if not written_header[kind]:
                writers[kind].writeheader()
        writers[kind].writerow(row)
        files[kind].flush()

    teacher = M.load_teacher()
    for scene in scenes:
        scene_data = get_scene_data(scene)
        sc = manifest["scenes"][scene]
        caches = []
        for pair in sc["probe_pairs"]:
            images8, _ = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            caches.append(M.cache_teacher_pair(teacher, images8, (ph, pw)))

        for pi, pair in enumerate(sc["probe_pairs"]):
            cache = caches[pi]
            other = caches[1 - pi]
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"], (images4.shape[-2], images4.shape[-1]))

            rows, feats24_s, H0, HT = d1_for_pair(
                teacher, scene_data, pair, cache, other, images8, images4, gt4, (ph, pw))
            for r in rows:
                write("feature_interpolation.csv", dict(scene=scene, pair_id=pi, **r))

            cam = camera_endpoints(teacher, scene_data, pair, cache, images4)
            write("camera_endpoints.csv", dict(scene=scene, pair_id=pi, **cam))

            d2_rows, e0, losses0, grads, G = d2_for_pair(
                teacher, scene_data, pair, cache, images8, images4, gt4, (ph, pw), feats24_s, H0, HT)
            for r in d2_rows:
                write("feature_direction.csv", dict(scene=scene, pair_id=pi, **r))

            # patch audit: per-patch residuals vs teacher-local quality
            with torch.no_grad():
                d_s, _ = M.replay_depth_nograd(teacher, insert_patch_feats(feats24_s, H0), images4)
                d_s = d_s.squeeze(0).squeeze(-1).cpu().numpy()
                d_t = cache["depth4"].squeeze(0).squeeze(-1).cpu().numpy()
            pp_s = perpatch_logres2(d_s, gt4, (ph, pw))
            pp_t = perpatch_logres2(d_t, gt4, (ph, pw))
            for r in patch_audit(H0, HT, grads, pp_s, pp_t):
                write("patch_audit.csv", dict(scene=scene, pair_id=pi, **r))

            print(f"[{scene} pair{pi}] E0={e0:.4f} teacherE={rows[4]['e_depth']:.4f} "
                  f"losses0={ {k: round(v, 4) for k, v in losses0.items()} } G={G:.2f}",
                  flush=True)
        del caches
        torch.cuda.empty_cache()

    for f in files.values():
        f.close()
    print("DONE")


if __name__ == "__main__":
    main()
