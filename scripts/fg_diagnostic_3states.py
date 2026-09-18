#!/usr/bin/env python3
"""3-state controlled diagnostic on fixed probe pairs (no new training).

States:
  θ₀     = frozen baseline (LoRA-B=0)
  θ_base = trained WITHOUT rel (rkdc1h)
  θ_rel  = trained WITH rel (rkdc1hr)

For each probe pair × state × {clean, masked}:
  Compute rel_rot, rel_tdir, maskdistill, rkd_sh_d/a, couple

For each probe pair × state (clean only):
  Compute gradient norms and cos(g_rel, g_base) where:
    g_base = ∇θ(L_feat + 1.5·L_rkd + L_couple)
    g_R    = ∇θ(L_rel_rot)  (rotation only)
    g_t    = ∇θ(L_rel_tdir) (translation direction only)

Usage: fg_diagnostic_3states.py --dataset scannetpp --scenes 09c1414f1b 1ada7a0617 ...
Requires: workspace/da3_{ds}_rkdcr/ckpts/ and workspace/da3_{base_dir}/ckpts/ to exist.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))

import common as fg
from depth_anything_3.test_time_adaption import protocol_v1 as P


def load_student_with_ckpt(student, ckpt_path, device):
    """Load LoRA weights into a fresh student."""
    student.load_lora_weights(ckpt_path)
    student.to(device)
    student.eval()
    return student


def compute_components(student, teacher_cache, images4, patch_hw, head_norm):
    """Forward student and compute all loss components (no backward)."""
    with torch.no_grad():
        tap_feats, ext_s, depth_s = P.student_forward_c2m(student, images4, with_depth=True)
        device = images4.device
        # maskdistill (on all positions, using teacher conf as weight)
        w = P.teacher_patch_conf(teacher_cache["conf4"].to(device), patch_hw)
        w = w / w.mean().clamp_min(1e-8)
        md = 0.0
        for layer in P.TAP_LAYERS:
            hs = head_norm(tap_feats[layer].float())
            ht = head_norm(teacher_cache["feats"][layer].float().to(device)).detach()
            huber = F.smooth_l1_loss(hs, ht, beta=1.0, reduction="none").mean(-1)
            cos = F.cosine_similarity(hs, ht, dim=-1)
            md += ((huber * w).mean() + 2.0 * (1.0 - (cos * w).mean()))
        md /= len(P.TAP_LAYERS)
        # rkd
        ext_t = teacher_cache["ext4"].to(device)
        rkd_loss, rkd_extra = P.loss_rkd_shared_pose_huber_w2c(ext_s, ext_t)
        d_val = rkd_extra["rkd_sh_d"]
        a_val = rkd_extra["rkd_sh_a"]
        # couple
        cp = P.loss_couple_w2c(ext_s, depth_s, ext_t,
                               teacher_cache["depth4"].to(device),
                               teacher_cache["conf4"].to(device))
        # rel (corrected formula)
        rel = P.loss_pose_rel(ext_s, ext_t)
    return {"maskdistill": float(md), "rkd_sh_d": d_val, "rkd_sh_a": a_val,
            "couple": float(cp[0]), "rel_rot": rel[1]["rel_rot"], "rel_tdir": rel[1]["rel_tdir"]}


def rkd_extra_val(loss, part):
    """Extract scalar from rkd loss output."""
    if isinstance(loss, torch.Tensor):
        return float(loss)
    return float(loss)


def compute_gradients(student, teacher_cache, images4, patch_hw, head_norm, params):
    """Compute gradient norms and cosines for each loss component separately."""
    device = images4.device
    ext_t = teacher_cache["ext4"].to(device)
    results = {}

    def get_flat_grads():
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=device))
        return torch.cat(grads)

    def clear_grads():
        for p in params:
            p.grad = None

    # Forward once
    tap_feats, ext_s, depth_s = P.student_forward_c2m(student, images4, with_depth=True)

    # g_R: gradient of rel_rot only
    clear_grads()
    rel_out = P.loss_pose_rel(ext_s, ext_t)
    # We need to separate rot and tdir - recompute manually
    if ext_s.dim() == 4: ext_s_2d = ext_s[0]
    else: ext_s_2d = ext_s
    if ext_t.dim() == 4: ext_t_2d = ext_t[0]
    else: ext_t_2d = ext_t
    R_s, t_s = ext_s_2d[..., :3, :3], ext_s_2d[..., :3, 3]
    R_t, t_t = ext_t_2d[..., :3, :3], ext_t_2d[..., :3, 3]
    S = R_s.shape[0]
    rot_l = 0.0; tdir_l = 0.0; np_ = 0
    for i in range(S):
        for j in range(i+1, S):
            Rr_s = R_s[i] @ R_s[j].transpose(-1, -2)
            Rr_t = R_t[i] @ R_t[j].transpose(-1, -2)
            tr_s = t_s[i].unsqueeze(-1) - Rr_s @ t_s[j].unsqueeze(-1)
            tr_t = t_t[i].unsqueeze(-1) - Rr_t @ t_t[j].unsqueeze(-1)
            rot_l = rot_l + ((Rr_s - Rr_t)**2).sum()
            tn_s = F.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
            tn_t = F.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
            tdir_l = tdir_l + (1.0 - (tn_s * tn_t).sum())
            np_ += 1
    rot_l = rot_l / np_; tdir_l = tdir_l / np_

    # g_R (rotation gradient)
    clear_grads()
    rot_l.backward(retain_graph=True)
    g_R = get_flat_grads()
    results["||g_R||"] = float(g_R.norm())

    # g_t (translation direction gradient)
    clear_grads()
    tdir_l.backward(retain_graph=True)
    g_t = get_flat_grads()
    results["||g_t||"] = float(g_t.norm())

    # g_base: gradient of (feat + 1.5*rkd + couple)
    clear_grads()
    w = P.teacher_patch_conf(teacher_cache["conf4"].to(device), patch_hw)
    w = w / w.mean().clamp_min(1e-8)
    md = 0.0
    for layer in P.TAP_LAYERS:
        hs = head_norm(tap_feats[layer].float())
        ht = head_norm(teacher_cache["feats"][layer].float().to(device)).detach()
        huber = F.smooth_l1_loss(hs, ht, beta=1.0, reduction="none").mean(-1)
        cos = F.cosine_similarity(hs, ht, dim=-1)
        md += ((huber * w).mean() + 2.0 * (1.0 - (cos * w).mean()))
    md /= len(P.TAP_LAYERS)
    rkd = P.loss_rkd_shared_pose_huber_w2c(ext_s, ext_t)[0]
    cp = P.loss_couple_w2c(ext_s, depth_s, ext_t,
                           teacher_cache["depth4"].to(device),
                           teacher_cache["conf4"].to(device))[0]
    base_loss = md + 1.5 * rkd + 1.0 * cp
    base_loss.backward()
    g_base = get_flat_grads()
    results["||g_base||"] = float(g_base.norm())

    # Cosines
    results["cos(g_R, g_base)"] = float(F.cosine_similarity(g_R.unsqueeze(0),
                                                             g_base.unsqueeze(0)))
    results["cos(g_t, g_base)"] = float(F.cosine_similarity(g_t.unsqueeze(0),
                                                             g_base.unsqueeze(0)))
    clear_grads()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--rel_dir", required=True, help="dir with rel-trained ckpts")
    ap.add_argument("--base_dir", required=True, help="dir with no-rel-trained ckpts")
    ap.add_argument("--n_probes", type=int, default=2)
    args = ap.parse_args()

    device = "cuda"
    fg.set_dataset(args.dataset)
    model_name = "model_weights/DA3-GIANT-1.1"

    all_results = {}

    teacher = P.create_teacher(model_name, device)
    student = P.create_student(model_name, device)

    for scene in args.scenes:
        print(f"\n=== {scene} ===", flush=True)
        data = fg.get_scene_data(scene)
        files = list(data.image_files)
        proto = P.build_scene_protocol(files, scene, dataset=args.dataset,
                                       n_train=10, n_shared=4, teacher_N=8)
        probe_pairs = proto["probe_pairs"][:args.n_probes]

        # Cache teacher for probe pairs
        from depth_anything_3.test_time_adaption.protocol_v1 import load_images_da3, PATCH_SIZE
        imgs_all = load_images_da3(files)
        probe_caches = []
        probe_images4 = []
        for pair in probe_pairs:
            imgs_t = torch.stack([imgs_all[i] for i in pair["teacher_frames"]]) \
                .unsqueeze(0).to(device)
            slots = [0, 2, 4, 6]
            # Compute actual patch_hw from image size (not hardcoded)
            ph = imgs_t.shape[-2] // PATCH_SIZE
            pw = imgs_t.shape[-1] // PATCH_SIZE
            cache = P.cache_teacher_pair(teacher, imgs_t, slots, (ph, pw))
            probe_caches.append(cache)
            probe_images4.append(imgs_t[0, slots].cpu())
            del imgs_t
        # Use actual patch_hw from cache conf shape
        conf_shape = probe_caches[0]["conf4"].shape
        patch_hw = (conf_shape[-2] // PATCH_SIZE, conf_shape[-1] // PATCH_SIZE)
        head_norm = student.da3.model.head.norm

        # Define 3 states
        states = {
            "theta_0 (frozen)": None,  # LoRA-B=0 (just reset)
            "theta_base (no rel)": os.path.join(args.base_dir, "ckpts", scene, "c2m_final_lora.pt"),
            "theta_rel (with rel)": os.path.join(args.rel_dir, "ckpts", scene, "c2m_final_lora.pt"),
        }

        scene_results = {}
        for state_name, ckpt in states.items():
            # Reset or load
            P.reset_lora_(student)
            student.to(device)
            if ckpt and os.path.exists(ckpt):
                student.load_lora_weights(ckpt)
                student.to(device)
            student.eval()

            state_res = {"clean": [], "masked": [], "grads": []}
            for pi, (cache, imgs4) in enumerate(zip(probe_caches, probe_images4)):
                imgs4 = imgs4.unsqueeze(0).to(device)
                # Clean forward
                clean = compute_components(student, cache, imgs4, patch_hw, head_norm)
                state_res["clean"].append(clean)
                # Masked forward (same mask seed as training)
                gen = torch.Generator(device=device).manual_seed(
                    P.stable_seed("mask", scene, 99, pi, 0))
                imgs_m, _ = P.mask_image_blocks(imgs4, 0.5, patch_hw, gen)
                masked = compute_components(student, cache, imgs_m, patch_hw, head_norm)
                state_res["masked"].append(masked)
                del imgs4, imgs_m

            # Gradient analysis (clean input, first probe only)
            if len(params_cache := student.get_trainable_params()) > 0:
                imgs4 = probe_images4[0].unsqueeze(0).to(device)
                grads = compute_gradients(student, probe_caches[0], imgs4,
                                          patch_hw, head_norm, params_cache)
                state_res["grads"].append(grads)
                del imgs4

            scene_results[state_name] = state_res
            print(f"  {state_name}: done", flush=True)

        all_results[scene] = scene_results
        torch.cuda.empty_cache()

    # ---- Report ----
    print("\n" + "=" * 90)
    print("DIAGNOSTIC TABLE (probe-pair means)")
    print("=" * 90)

    for scene, sres in all_results.items():
        print(f"\n### {scene}")
        print(f"{'metric':22s} | {'θ₀ (frozen)':>12s} | {'θ_base':>12s} | {'θ_rel':>12s}")
        print("-" * 70)

        for cond in ("clean", "masked"):
            print(f"  [{cond}]")
            for comp in ("rel_rot", "rel_tdir", "maskdistill", "rkd_sh_d", "couple"):
                vals = []
                for sn in ("theta_0 (frozen)", "theta_base (no rel)", "theta_rel (with rel)"):
                    v = np.mean([d[comp] for d in sres[sn][cond]])
                    vals.append(v)
                print(f"    {comp:18s} | {vals[0]:12.4f} | {vals[1]:12.4f} | {vals[2]:12.4f}")

        print(f"  [gradients (clean, probe 0)]")
        for g in ("||g_R||", "||g_t||", "||g_base||", "cos(g_R, g_base)", "cos(g_t, g_base)"):
            vals = []
            for sn in ("theta_0 (frozen)", "theta_base (no rel)", "theta_rel (with rel)"):
                v = sres[sn]["grads"][0][g] if sres[sn]["grads"] else np.nan
                vals.append(v)
            print(f"    {g:20s} | {vals[0]:12.4f} | {vals[1]:12.4f} | {vals[2]:12.4f}")

    # Save JSON
    out_path = os.path.join(ROOT, "workspace", "fg_diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
