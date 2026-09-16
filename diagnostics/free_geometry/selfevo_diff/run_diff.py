#!/usr/bin/env python3
"""SELFEVO_DIFF: differential check of the C2M_ABS absolute-pose pseudo-label loss.

For each scene (train_pairs[0] of the final_protocol scannetpp manifest):
  teacher = frozen VGGT-1B, one 16-view fp32 forward (cache built exactly as
            train_arms.py does via M.cache_teacher_pair);
  student = same frozen VGGT-1B (no LoRA), 4-view forward of the shared frames,
            unmasked and with the training-identical 50% block mask
            (mask_image_blocks + stable_seed("mask", scene, 0, 0, 0)).

Loss A: diagnostics/free_geometry/abs_pose_loss.py::loss_abs_pose_norm (ours).
Loss B: SelfEvo's original chain, loaded BY FILE from /root/autodl-tmp/SelfEvo
        (read-only): build_seq_from_indices_batch (gather shared slots, point
        masks from teacher depth/conf, prune_ratio=0.05) ->
        lift_depth_to_cam_world_points_torch ->
        normalize_camera_extrinsics_and_points_batch_gpu (scale_by_points=True)
        -> compute_camera_loss / camera_loss_single (l1, weights 1/1/0.5).
        A B-cache variant feeds our cache's depth4/conf4 (4-view replay)
        instead of the gathered 16-view teacher depth/conf, isolating the
        depth-source factor.
Gauge decomposition: mean per-frame translation L1 between the student's raw
pose_enc T and (1) teacher raw T, (2) our normalized teacher target,
(3) both sides self-normalized (our rule and SelfEvo's rule).

Inference + loss only; nothing is trained. Writes SELFEVO_DIFF.{json,md} next
to the scene manifest.
"""

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FG_DIR = os.path.dirname(HERE)
sys.path.insert(0, FG_DIR)

import common  # noqa: E402
from common import STUDENT_INDICES, get_scene_data, load_manifest, stable_seed  # noqa: E402
import modeling as M  # noqa: E402
from abs_pose_loss import loss_abs_pose_norm, _teacher_target  # noqa: E402
from train_arms import mask_image_blocks  # noqa: E402
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri  # noqa: E402

SE_ROOT = "/root/autodl-tmp/SelfEvo"
IMAGE_HW = (378, 504)
SHARED = list(STUDENT_INDICES)  # [0,2,4,6] shared slots in the 16-view teacher
ABS_POSE_W = 5.0                # SelfEvo config default.yaml loss.camera.weight
TOL = 1e-4


def load_selfevo_modules():
    """Import-by-file of SelfEvo's loss/normalization/frame_sampling modules.

    Import isolation: a synthetic top-level `train_utils` package points at
    SelfEvo/training/train_utils so their `from train_utils.general import ...`
    resolves to SelfEvo's own file. Their `vggt.*` imports resolve to this
    repo's src/vggt - utils/{pose_enc,geometry,rotation,helper}.py verified
    byte-identical to SelfEvo's via diff beforehand."""
    pkg = types.ModuleType("train_utils")
    pkg.__path__ = [os.path.join(SE_ROOT, "training", "train_utils")]
    sys.modules["train_utils"] = pkg

    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(SE_ROOT, rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    return {
        "general": _load("train_utils.general", "training/train_utils/general.py"),
        "norm": _load("se_train_utils_normalization", "training/train_utils/normalization.py"),
        "loss": _load("se_training_loss", "training/loss.py"),
        "fs": _load("se_train_utils_frame_sampling", "training/train_utils/frame_sampling.py"),
    }


def make_point_mask_from_conf(depth, depth_conf, prune_ratio):
    """Verbatim replica of SelfEvo trainer.py:799-819 _make_point_mask_from_conf."""
    mask_depth = torch.isfinite(depth) & (depth > 0)
    if depth_conf is None or (not torch.is_tensor(depth_conf)) or prune_ratio <= 0:
        return mask_depth
    B, L, H, W = depth_conf.shape
    conf_flat = depth_conf.reshape(B, -1)
    try:
        q = torch.nanquantile(conf_flat, prune_ratio, dim=-1, keepdim=True)
    except Exception:
        q = torch.quantile(torch.nan_to_num(conf_flat, nan=float("-inf")),
                           prune_ratio, dim=-1, keepdim=True)
    mask_conf = (conf_flat >= q).view(B, L, H, W)
    return (mask_depth & mask_conf).bool(), float(q[0, 0])


@torch.no_grad()
def student_forward(vggt, images):
    """M.student_preds equivalent on the plain frozen VGGT (no LoRA wrapper):
    aggregator under autocast (training convention), heads in fp32."""
    with torch.autocast(device_type="cuda", enabled=True):
        feats24, psi = vggt.aggregator(images)
    assert psi == common.PATCH_START_IDX
    with torch.autocast(device_type="cuda", enabled=False):
        feats24_f = [t.float() for t in feats24]
        depth, conf = vggt.depth_head(feats24_f, images=images, patch_start_idx=psi)
        pose_list = vggt.camera_head(feats24_f)
    return {"depth": depth, "conf": conf, "pose_enc": pose_list[-1]}


@torch.no_grad()
def se_normalize(se, E, K, depths, point_masks):
    """SelfEvo trainer.py:1934-1951 block. Returns (new_E, new_D, avg_scale)."""
    cam_pts, world_pts, _ = se["norm"].lift_depth_to_cam_world_points_torch(
        depths=depths, extrinsics=E, intrinsics=K)
    # avg_scale exactly as in normalize_camera_extrinsics_and_points_batch_gpu
    R0, t0 = E[:, 0, :3, :3], E[:, 0, :3, 3]
    nwp = (world_pts @ R0.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + \
        t0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    dist = nwp.norm(dim=-1)
    pm = point_masks
    avg_scale = ((dist * pm).sum(dim=[1, 2, 3])
                 / (pm.sum(dim=[1, 2, 3]) + 1e-3)).clamp(min=1e-6, max=1e6)
    new_E, _, _, new_D = se["norm"].normalize_camera_extrinsics_and_points_batch_gpu(
        extrinsics=E, cam_points=cam_pts, world_points=world_pts, depths=depths,
        point_masks=point_masks, scale_by_points=True)
    return new_E, new_D, avg_scale


@torch.no_grad()
def selfevo_chain(se, pose_enc_s, pose_enc16, depth16, conf16, images16):
    """Primary Loss B: SelfEvo's own build_seq_from_indices_batch gather +
    normalize + compute_camera_loss. Returns (result dict, gt_pose_encoding)."""
    idx = torch.tensor([SHARED], device=images16.device)
    t_out = {"pose_enc": pose_enc16, "depth": depth16, "depth_conf": conf16}
    chunk = se["fs"].build_seq_from_indices_batch(
        t_out, {"images": images16}, idx, prune_ratio=0.05)
    E, K, D, pmask = chunk["extrinsics"], chunk["intrinsics"], chunk["depths"], chunk["point_masks"]
    new_E, _, avg_scale = se_normalize(se, E, K, D, pmask)
    pred = {"pose_enc_list": [pose_enc_s]}
    batch = {"point_masks": pmask, "extrinsics": new_E, "intrinsics": K,
             "images": chunk["images"]}
    out = se["loss"].compute_camera_loss(pred, batch)
    gt_enc = extri_intri_to_pose_encoding(new_E, K, IMAGE_HW,
                                          pose_encoding_type="absT_quaR_FoV")
    res = {"loss_T": float(out["loss_T"]), "loss_R": float(out["loss_R"]),
           "loss_FL": float(out["loss_FL"]), "loss_camera": float(out["loss_camera"]),
           "x5": float(5.0 * out["loss_camera"]),
           "avg_scale": float(avg_scale[0]),
           "mask_valid_per_frame": [int(v) for v in pmask[0].sum(dim=[-1, -2])]}
    return res, gt_enc


@torch.no_grad()
def selfevo_chain_cache_depth(se, pose_enc_s, pose_enc16, cache, images16):
    """B-cache variant: SelfEvo's math but fed with our cache's depth4/conf4
    (frozen DPT replay on the 4 shared slots) instead of gathered 16v depth."""
    E4, K4 = pose_encoding_to_extri_intri(pose_enc16[:, SHARED].float(), IMAGE_HW)
    D = cache["depth4"].squeeze(-1).float()           # [1,4,H,W]
    C = cache["conf4"].float()
    if C.dim() == 5:
        C = C.squeeze(2)
    pmask, q = make_point_mask_from_conf(D, C, 0.05)
    new_E, _, avg_scale = se_normalize(se, E4, K4, D, pmask)
    pred = {"pose_enc_list": [pose_enc_s]}
    batch = {"point_masks": pmask, "extrinsics": new_E, "intrinsics": K4,
             "images": images16[:, SHARED]}
    out = se["loss"].compute_camera_loss(pred, batch)
    gt_enc = extri_intri_to_pose_encoding(new_E, K4, IMAGE_HW,
                                          pose_encoding_type="absT_quaR_FoV")
    res = {"loss_T": float(out["loss_T"]), "loss_R": float(out["loss_R"]),
           "loss_FL": float(out["loss_FL"]), "loss_camera": float(out["loss_camera"]),
           "x5": float(5.0 * out["loss_camera"]),
           "avg_scale": float(avg_scale[0]), "conf_q05": q,
           "mask_valid_per_frame": [int(v) for v in pmask[0].sum(dim=[-1, -2])]}
    return res, gt_enc


def quat_sign_stats(pose_enc_s, gt_enc):
    qs, qt = pose_enc_s[0, :, 3:7].float(), gt_enc[0, :, 3:7].float()
    dot = (qs * qt).sum(-1)
    flip = torch.where(dot.unsqueeze(-1) < 0, -torch.ones_like(qs), torch.ones_like(qs))
    r_flip = (qs * flip - qt).abs().mean()
    return {"quat_dot": [float(v) for v in dot],
            "n_negative_dot": int((dot < 0).sum()),
            "loss_R_if_sign_matched": float(r_flip)}


@torch.no_grad()
def gauge_decomposition(se, pose_enc_s, preds_s, cache, gt_enc_se):
    """Mean per-frame |dt| (pose_enc T components) under four comparisons."""
    t_s = pose_enc_s[0, :, :3].float()
    t_t_raw = cache["pose_enc8"][0, SHARED, :3].float()

    # our normalized teacher target (exactly what loss_abs_pose_norm uses)
    tgtA, scaleA = _teacher_target(cache["pose_enc8"].float(), cache["depth4"],
                                   cache["conf4"], SHARED, IMAGE_HW)
    t_tA = tgtA[0, :, :3]
    # student self-normalized with OUR rule (own depth/conf, own frame 0)
    tgtS, scaleS = _teacher_target(pose_enc_s.detach().float(), preds_s["depth"],
                                   preds_s["conf"], [0, 1, 2, 3], IMAGE_HW)
    t_sA = tgtS[0, :, :3]

    # both sides normalized with SELVEVO's rule; teacher side = gt_enc_se
    t_tSE = gt_enc_se[0, :, :3].float()
    E_s, K_s = pose_encoding_to_extri_intri(pose_enc_s.detach().float(), IMAGE_HW)
    D_s = preds_s["depth"].squeeze(-1).float()
    C_s = preds_s["conf"].float()
    if C_s.dim() == 5:
        C_s = C_s.squeeze(2)
    pmask_s, _ = make_point_mask_from_conf(D_s, C_s, 0.05)
    new_Es, _, scaleSE_s = se_normalize(se, E_s, K_s, D_s, pmask_s)
    gt_enc_s = extri_intri_to_pose_encoding(new_Es, K_s, IMAGE_HW,
                                            pose_encoding_type="absT_quaR_FoV")
    t_sSE = gt_enc_s[0, :, :3]

    def l1(a, b):
        d = (a - b).abs()
        return float(d.mean()), [float(v) for v in d.mean(dim=-1)]

    m_raw, pf_raw = l1(t_s, t_t_raw)
    m_tgtA, pf_tgtA = l1(t_s, t_tA)
    m_bothA, pf_bothA = l1(t_sA, t_tA)
    m_bothSE, pf_bothSE = l1(t_sSE, t_tSE)
    return {
        "L1_raw_vs_teacher_raw": m_raw,
        "L1_raw_vs_teacher_normA": m_tgtA,          # == Loss A abs_T
        "L1_both_normA": m_bothA,
        "L1_both_normSE": m_bothSE,
        "artifact_share_A": (m_tgtA - m_bothA) / m_tgtA if m_tgtA > 0 else float("nan"),
        "inflation_A": m_tgtA / m_bothA if m_bothA > 0 else float("nan"),
        "artifact_share_SE": (m_tgtA - m_bothSE) / m_tgtA if m_tgtA > 0 else float("nan"),
        "inflation_SE": m_tgtA / m_bothSE if m_bothSE > 0 else float("nan"),
        "scale_A_teacher": float(scaleA), "scale_A_student": float(scaleS),
        "scale_SE_student": float(scaleSE_s[0]),
        "t_norm_student_raw": float(t_s.norm(dim=-1).mean()),
        "t_norm_teacher_raw": float(t_t_raw.norm(dim=-1).mean()),
        "t_norm_teacher_normA": float(t_tA.norm(dim=-1).mean()),
        "per_frame": {"raw_vs_raw": pf_raw, "raw_vs_tgtA": pf_tgtA,
                      "both_normA": pf_bothA, "both_normSE": pf_bothSE},
        "vectors": {
            "student_raw_T": np.round(t_s.cpu().numpy(), 6).tolist(),
            "teacher_raw_T": np.round(t_t_raw.cpu().numpy(), 6).tolist(),
            "teacher_normA_T": np.round(t_tA.cpu().numpy(), 6).tolist(),
            "student_normA_T": np.round(t_sA.cpu().numpy(), 6).tolist(),
            "teacher_normSE_T": np.round(t_tSE.cpu().numpy(), 6).tolist(),
            "student_normSE_T": np.round(t_sSE.cpu().numpy(), 6).tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        common._REPO_ROOT, "artifacts", "diagnostics", "final_protocol",
        "scannetpp", "scene_manifest.json"))
    ap.add_argument("--scenes", nargs="*", default=["09c1414f1b", "1ada7a0617"])
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    out_dir = os.path.dirname(args.manifest)
    args.out_md = args.out_md or os.path.join(out_dir, "SELFEVO_DIFF.md")
    args.out_json = args.out_json or os.path.join(out_dir, "SELFEVO_DIFF.json")

    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[gpu] free={free_b / 2**30:.1f}GiB total={total_b / 2**30:.1f}GiB", flush=True)
    assert free_b >= 20 * 2**30, "need >=20GiB free GPU memory"

    manifest = load_manifest(args.manifest)
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    se = load_selfevo_modules()
    teacher = M.load_teacher("cuda")

    results = {}
    for scene in args.scenes:
        pair = manifest["scenes"][scene]["train_pairs"][0]
        assert pair["student_frames"] == [pair["teacher_frames"][i] for i in SHARED]
        scene_data = get_scene_data(scene)
        images16, images4 = M.load_pair_images(scene_data, pair["teacher_frames"], "cuda")
        ph, pw = images16.shape[-2] // 14, images16.shape[-1] // 14
        patch_hw = (ph, pw)
        assert (images16.shape[-2], images16.shape[-1]) == IMAGE_HW

        # ---- teacher 16v forward + train_arms-identical cache ----
        cache = M.cache_teacher_pair(teacher, images16, patch_hw)
        pose_enc16 = cache["pose_enc8"]                      # [1,16,9]
        # full-context 16v depth/conf (SelfEvo-faithful teacher outputs)
        depth16, conf16 = M.replay_depth_nograd(
            teacher, M.build_replay_list(cache["feats"]), images16,
            psi=common.PATCH_START_IDX)
        # teacher depth/conf on shared slots: 16v-gathered vs 4v-replay delta
        d_gather = depth16[:, SHARED].squeeze(-1)
        d_replay = cache["depth4"].squeeze(-1)
        rel_d = ((d_gather - d_replay).abs() / d_gather.clamp_min(1e-6))
        depth_src = {"mean_rel_abs_diff_depth": float(rel_d.mean()),
                     "conf_gather_mean": float(conf16[:, SHARED].mean()),
                     "conf_replay_mean": float(cache["conf4"].mean())}

        srec = {"pair": pair, "patch_hw": patch_hw, "depth_source": depth_src,
                "variants": {}}

        gen = torch.Generator(device=images4.device).manual_seed(
            stable_seed("mask", scene, 0, 0, 0))
        images4_m, pmask = mask_image_blocks(images4, 0.5, patch_hw, gen)
        for variant, imgs in (("unmasked", images4), ("masked50", images4_m)):
            preds_s = student_forward(teacher, imgs)
            pose_s = preds_s["pose_enc"].float()

            # Loss A (ours)
            lossA, extraA = loss_abs_pose_norm(pose_s, cache)
            rec = {"A_ours": {"abs_T": extraA["abs_T"], "abs_R": extraA["abs_R"],
                              "abs_FL": extraA["abs_FL"], "abs_scale": extraA["abs_scale"],
                              "total": float(lossA), "x5": float(ABS_POSE_W * lossA)}}

            # Loss B (SelfEvo original chain, 16v-gathered depth/conf)
            resB, gt_enc_se = selfevo_chain(se, pose_s, pose_enc16, depth16, conf16, images16)
            rec["B_selfevo"] = resB
            rec["B_selfevo"]["quat_sign"] = quat_sign_stats(pose_s, gt_enc_se)

            # Loss B with cache depth4/conf4 (depth-source isolation)
            resBc, _ = selfevo_chain_cache_depth(se, pose_s, pose_enc16, cache, images16)
            rec["B_selfevo_cache_depth"] = resBc

            # A-vs-B deltas
            rec["delta_A_minus_B"] = {
                k: rec["A_ours"][a] - resB[b]
                for k, a, b in [("T", "abs_T", "loss_T"), ("R", "abs_R", "loss_R"),
                                ("FL", "abs_FL", "loss_FL"),
                                ("total", "total", "loss_camera"), ("x5", "x5", "x5")]}

            # gauge decomposition
            rec["gauge"] = gauge_decomposition(se, pose_s, preds_s, cache, gt_enc_se)

            # raw student pose_enc dump
            rec["student_pose_enc_raw"] = np.round(pose_s[0].cpu().numpy(), 6).tolist()
            srec["variants"][variant] = rec
            print(f"[{scene}] {variant}: A(T={extraA['abs_T']:.4f} R={extraA['abs_R']:.4f} "
                  f"FL={extraA['abs_FL']:.4f} x5={5.0 * float(lossA):.4f}) | "
                  f"B(T={resB['loss_T']:.4f} R={resB['loss_R']:.4f} FL={resB['loss_FL']:.4f} "
                  f"x5={resB['x5']:.4f})", flush=True)
        srec["mask_ratio_actual"] = float(pmask.mean())
        results[scene] = srec
        del cache, depth16, conf16, images16, images4, images4_m
        torch.cuda.empty_cache()

    payload = {"provenance": {
        "vggt_utils_diff": "SelfEvo/vggt/utils/{pose_enc,geometry,rotation,helper}.py "
                           "byte-identical to src/vggt/vggt/utils/ (diff -q)",
        "loss_py_diff": "SelfEvo/training/loss.py vs src/vggt/vggt/training/loss.py: "
                        "only 6 commented-out lines (EXP1/EXP2) differ; "
                        "compute_camera_loss/camera_loss_single identical",
        "general_py_diff": "SelfEvo/training/train_utils/general.py differs only by "
                           "extra helpers (_gather_BS/_minmax_norm/_unwrap_module); "
                           "check_and_fix_inf_nan identical",
        "normalization_py": "SelfEvo-only file (this repo has normalize_batch.py "
                            "instead); loaded by file path",
        "module_loading": "importlib.spec_from_file_location with synthetic "
                          "top-level train_utils package -> SelfEvo/training/train_utils"},
        "weights": "1/1/0.5 (compute_camera_loss defaults), x5 = loss.camera.weight "
                   "from SelfEvo config/default.yaml:113 == train_arms.ABS_POSE_W",
        "results": results}

    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=1)
    write_md(args.out_md, payload)
    print(f"wrote {args.out_md} and {args.out_json}", flush=True)


def write_md(path, payload):
    L = []
    L.append("# SELFEVO_DIFF - C2M_ABS camera-loss differential check\n")
    L.append("Inference-only comparison on identical frozen VGGT-1B forwards "
             "(teacher 16v fp32, student 4v; scenes' train_pairs[0], 378x504). "
             "No training, no eval.\n")
    L.append("## Module provenance (import isolation)\n")
    for k, v in payload["provenance"].items():
        L.append(f"- **{k}**: {v}")
    L.append(f"\nWeights: {payload['weights']}\n")
    L.append("## Loss values per scene x variant x implementation\n")
    L.append("| scene | variant | impl | T | R | FL | total(1/1/0.5) | x5 | scale |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for scene, srec in payload["results"].items():
        for variant, rec in srec["variants"].items():
            a, b, bc = rec["A_ours"], rec["B_selfevo"], rec["B_selfevo_cache_depth"]
            L.append(f"| {scene} | {variant} | A (ours) | {a['abs_T']:.6f} | "
                     f"{a['abs_R']:.6f} | {a['abs_FL']:.6f} | {a['total']:.6f} | "
                     f"{a['x5']:.6f} | {a['abs_scale']:.4f} |")
            L.append(f"| {scene} | {variant} | B (SelfEvo, 16v depth) | {b['loss_T']:.6f} | "
                     f"{b['loss_R']:.6f} | {b['loss_FL']:.6f} | {b['loss_camera']:.6f} | "
                     f"{b['x5']:.6f} | {b['avg_scale']:.4f} |")
            L.append(f"| {scene} | {variant} | B (SelfEvo, cache depth4) | {bc['loss_T']:.6f} | "
                     f"{bc['loss_R']:.6f} | {bc['loss_FL']:.6f} | {bc['loss_camera']:.6f} | "
                     f"{bc['x5']:.6f} | {bc['avg_scale']:.4f} |")
    L.append("\n### A-vs-B agreement (tolerance 1e-4)\n")
    L.append("| scene | variant | dT | dR | dFL | d(total) | agree? | R if sign-matched (B) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for scene, srec in payload["results"].items():
        for variant, rec in srec["variants"].items():
            d = rec["delta_A_minus_B"]
            ok = all(abs(d[k]) <= TOL for k in ("T", "R", "FL"))
            L.append(f"| {scene} | {variant} | {d['T']:+.2e} | {d['R']:+.2e} | "
                     f"{d['FL']:+.2e} | {d['total']:+.2e} | {'YES' if ok else 'NO'} | "
                     f"{rec['B_selfevo']['quat_sign']['loss_R_if_sign_matched']:.6f} |")
    L.append("\n### Teacher depth source (16v gather vs cache 4v replay)\n")
    L.append("| scene | mean rel abs depth diff | conf mean (16v gather) | conf mean (4v replay) |")
    L.append("|---|---|---|---|")
    for scene, srec in payload["results"].items():
        ds = srec["depth_source"]
        L.append(f"| {scene} | {ds['mean_rel_abs_diff_depth']:.4f} | "
                 f"{ds['conf_gather_mean']:.4f} | {ds['conf_replay_mean']:.4f} |")
    L.append("\n## Gauge decomposition (mean per-frame translation L1)\n")
    L.append("| scene | variant | raw vs raw | raw vs tgtA (=abs_T) | both normA | "
             "both normSE | artifact share | inflation | scaleA_t | scaleA_s | scaleSE_s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for scene, srec in payload["results"].items():
        for variant, rec in srec["variants"].items():
            g = rec["gauge"]
            L.append(f"| {scene} | {variant} | {g['L1_raw_vs_teacher_raw']:.4f} | "
                     f"{g['L1_raw_vs_teacher_normA']:.4f} | {g['L1_both_normA']:.4f} | "
                     f"{g['L1_both_normSE']:.4f} | {g['artifact_share_A']:.1%} | "
                     f"{g['inflation_A']:.2f}x | {g['scale_A_teacher']:.4f} | "
                     f"{g['scale_A_student']:.4f} | {g['scale_SE_student']:.4f} |")
    L.append("\n`raw vs raw` = student raw T vs teacher 16v raw T at shared slots; "
             "`raw vs tgtA` = our loss target (identical to abs_T); `both normA` = "
             "each side self-normalized by our rule; `both normSE` = each side "
             "self-normalized by SelfEvo's rule. artifact share = "
             "(raw_vs_tgtA - both_normA)/raw_vs_tgtA.\n")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
