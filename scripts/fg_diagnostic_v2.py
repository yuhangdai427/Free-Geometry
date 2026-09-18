#!/usr/bin/env python3
"""3-state controlled diagnostic v2 — fixes every audit gap of v1 (2026-09-18).

States (per scene):
  theta0    = frozen baseline; LoRA re-init with the TRAINING seed
              (torch.manual_seed(stable_seed("lora_init", scene, seed)) ->
               reset_lora_), so theta0 == training step-0 exactly.
  theta_base = trained WITHOUT rel (arm rkdc1h), ckpt existence ASSERTED
              (+ sha256 recorded; silent fallback removed).
  theta_rel  = trained WITH corrected rel (arm rkdc1hr), same assertions.

Measurements per scene:
  [M1] loss components + per-camera-pair decomposition, PER INPUT (probe0,
       probe1, and optional train pairs), PER CONDITION (clean + 4 fixed
       masks). No probe averaging. Per camera pair (i,j): chordal rot loss,
       1-cos tdir loss, geodesic angle (deg), tdir angle (deg), ||t_rel||
       for teacher & student, R orthogonality error, det(R), depth validity.
  [M2] per-component gradients (6 separate backwards, one shared forward):
       g_md, g_rkd_d, g_rkd_a, g_cp, g_R, g_t — norms + full 6x6 cosine
       matrix + weighted g_base row. For EVERY probe (0 and 1) and BOTH
       clean and masked input. Split by parameter group: lora_early
       (block<20), lora_late (block>=20), cam_token — group-level norms and
       key cosines, so one dominant group can no longer masquerade as
       "global".
  [M3] Check C — counterfactual ONE AdamW step (fresh optimizer, training
       hyperparams) from theta_base / theta_rel on a fixed masked input:
       4 branches (base / +R / +t / +R+t), then re-measure all 6 components
       on the update input AND on probe0 clean.

Output: full-precision JSON (no rounding) + console summary.
"""
import argparse, hashlib, json, math, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/autodl-tmp/Free-Geometry"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))

import common as fg
from depth_anything_3.test_time_adaption import protocol_v1 as P
from depth_anything_3.test_time_adaption.protocol_v1 import load_images_da3, PATCH_SIZE

N_FIXED_MASKS = 4
COMPONENTS = ["maskdistill", "rkd_sh_d", "rkd_sh_a", "couple", "rel_rot", "rel_tdir"]


# --------------------------------------------------------------------------- helpers
def sha256_of(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_buf):
            h.update(chunk)
    return h.hexdigest()


def classify_param_groups(student):
    """-> {group_name: [(param_name, param), ...]} for trainable params."""
    groups = {"lora_early": [], "lora_late": [], "cam_token": [], "other": []}
    for name, p in student.da3.named_parameters():
        if not p.requires_grad:
            continue
        if "camera_token" in name:
            groups["cam_token"].append((name, p))
        elif "lora_" in name:
            m = re.search(r"blocks\.(\d+)\.", name)
            blk = int(m.group(1)) if m else -1
            groups["lora_early" if 0 <= blk < 20 else "lora_late"].append((name, p))
        else:
            groups["other"].append((name, p))
    return {k: v for k, v in groups.items() if v}


def rot_geodesic_deg(Ra, Rb):
    """geodesic angle between two rotations, degrees."""
    tr = torch.diagonal(Ra.transpose(-1, -2) @ Rb, dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))


def decompose_pairs(ext_s, ext_t, depth_s=None):
    """Per-camera-pair rel-pose decomposition (corrected formula), no grad.

    ext_*: [1,S,4,4] or [S,4,4] w2c. Returns (components, per_pair rows)."""
    if ext_s.dim() == 4:
        ext_s = ext_s[0]
    if ext_t.dim() == 4:
        ext_t = ext_t[0]
    R_s, t_s = ext_s[..., :3, :3].float(), ext_s[..., :3, 3].float()
    R_t, t_t = ext_t[..., :3, :3].float(), ext_t[..., :3, 3].float()
    S = R_s.shape[0]
    rows, rot_l, tdir_l = [], 0.0, 0.0
    for i in range(S):
        for j in range(i + 1, S):
            Rr_s = R_s[i] @ R_s[j].transpose(-1, -2)
            Rr_t = R_t[i] @ R_t[j].transpose(-1, -2)
            tr_s = t_s[i] - Rr_s @ t_s[j]
            tr_t = t_t[i] - Rr_t @ t_t[j]
            rot = ((Rr_s - Rr_t) ** 2).sum()
            tn_s = F.normalize(tr_s, dim=-1, eps=1e-8)
            tn_t = F.normalize(tr_t, dim=-1, eps=1e-8)
            tdir = 1.0 - (tn_s * tn_t).sum()
            ang = torch.rad2deg(torch.arccos(((tn_s * tn_t).sum() / 2 + 0.5).clamp(-1, 1)))
            rows.append({
                "ij": f"{i}{j}",
                "rot_chordal": float(rot), "tdir_1mc": float(tdir),
                "rot_geodesic_deg": float(rot_geodesic_deg(Rr_s, Rr_t)),
                "tdir_angle_deg": float(ang),
                "t_rel_norm_teacher": float(tr_t.norm()),
                "t_rel_norm_student": float(tr_s.norm()),
            })
            rot_l = rot_l + rot
            tdir_l = tdir_l + tdir
    n = len(rows)
    comps = {"rel_rot": float(rot_l / n), "rel_tdir": float(tdir_l / n)}
    # per-view sanity: R orthogonality error + det + depth validity
    for i in range(S):
        orth = (R_s[i].transpose(-1, -2) @ R_s[i] - torch.eye(3, device=R_s.device)).norm()
        rows_extra = rows[min(i, len(rows) - 1)]
        rows_extra[f"view{i}_orth_err"] = float(orth)
        rows_extra[f"view{i}_det"] = float(torch.det(R_s[i]))
    if depth_s is not None:
        d = depth_s.squeeze(0).squeeze(-1) if depth_s.dim() >= 4 else depth_s
        d = d.float()
        for i in range(min(S, d.shape[0])):
            di = d[i]
            rows[min(i, len(rows) - 1)][f"view{i}_depth_finite_frac"] = float(
                (torch.isfinite(di) & (di > 0)).float().mean())
    return comps, rows


def rkd_parts(ext_s, ext_t, delta=0.2):
    """d/a pieces of loss_rkd_shared_pose_huber_w2c as separate tensors."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(ext):
            if ext.dim() == 4:
                ext = ext[0]
            R, t = ext[..., :3, :3].float(), ext[..., :3, 3].float()
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)

        cs, ct = centers(ext_s), centers(ext_t).detach()
        S = cs.shape[0]
        pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
        ds = torch.stack([(cs[i] - cs[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dt = torch.stack([(ct[i] - ct[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dloss = F.huber_loss(ds / ds.mean().clamp_min(1e-8),
                             dt / dt.mean().clamp_min(1e-8), delta=delta)
        aterms = []
        for k in range(S):
            rest = [x for x in range(S) if x != k]
            for a, b in [(rest[0], rest[1]), (rest[0], rest[2]), (rest[1], rest[2])]:
                vs1 = F.normalize(cs[a] - cs[k], dim=-1, eps=1e-8)
                vs2 = F.normalize(cs[b] - cs[k], dim=-1, eps=1e-8)
                vt1 = F.normalize(ct[a] - ct[k], dim=-1, eps=1e-8)
                vt2 = F.normalize(ct[b] - ct[k], dim=-1, eps=1e-8)
                aterms.append(F.huber_loss((vs1 * vs2).sum(), (vt1 * vt2).sum(), delta=delta))
        aloss = torch.stack(aterms).mean()
    return dloss, aloss


def forward_all(student, cache, images4, patch_hw, head_norm):
    """One student forward -> dict of the 6 differentiable component tensors
    (teacher side detached) + tap feats handle for md."""
    tap_feats, ext_s, depth_s = P.student_forward_c2m(student, images4, with_depth=True)
    device = images4.device
    cache_d = {"feats": {l: t.to(device) for l, t in cache["feats"].items()},
               "conf4": cache["conf4"].to(device), "patch_hw": cache["patch_hw"]}
    ones = torch.ones(1, images4.shape[1], patch_hw[0] * patch_hw[1],
                      device=device)  # loss_all_pos mode, [1,S,P] as in training
    md, _ = P.loss_maskdistill(head_norm, cache_d, tap_feats, ones)
    ext_t = cache["ext4"].to(device)
    rd, ra = rkd_parts(ext_s, ext_t)
    cp, _ = P.loss_couple_w2c(ext_s, depth_s, ext_t,
                              cache["depth4"].to(device), cache["conf4"].to(device))
    def to4(x):  # student ext can be [1,S,4,4]; teacher cache ext4 is [S,3,4]
        return x.unsqueeze(0) if x.dim() == 3 else x
    e2d_s, e2d_t = to4(ext_s), to4(ext_t)
    R_s, t_s = e2d_s[..., :3, :3], e2d_s[..., :3, 3]
    R_t, t_t = e2d_t[..., :3, :3], e2d_t[..., :3, 3]
    S = R_s.shape[1]
    rot_l, tdir_l, np_ = 0.0, 0.0, 0
    for i in range(S):
        for j in range(i + 1, S):
            Rr_s = R_s[:, i] @ R_s[:, j].transpose(-1, -2)
            Rr_t = R_t[:, i] @ R_t[:, j].transpose(-1, -2)
            tr_s = t_s[:, i].unsqueeze(-1) - Rr_s @ t_s[:, j].unsqueeze(-1)
            tr_t = t_t[:, i].unsqueeze(-1) - Rr_t @ t_t[:, j].unsqueeze(-1)
            rot_l = rot_l + ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
            tn_s = F.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
            tn_t = F.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
            tdir_l = tdir_l + (1.0 - (tn_s * tn_t).sum(-1)).mean()
            np_ += 1
    rot_l, tdir_l = rot_l / np_, tdir_l / np_
    return {"maskdistill": md, "rkd_sh_d": rd, "rkd_sh_a": ra,
            "couple": cp, "rel_rot": rot_l, "rel_tdir": tdir_l}


def snap_grads(groups):
    """-> {group: flat grad vector or None}"""
    out = {}
    for g, plist in groups.items():
        vecs = []
        for _, p in plist:
            vecs.append(p.grad.detach().flatten() if p.grad is not None
                        else torch.zeros(p.numel()))
        out[g] = torch.cat(vecs)
    return out


def grad_matrix(student, cache, images4, patch_hw, head_norm, groups):
    """6 backwards on one shared forward -> per-group grads + cosine matrix."""
    device = images4.device
    comps = forward_all(student, cache, images4, patch_hw, head_norm)
    all_groups = dict(groups)
    all_groups["ALL"] = [(n, p) for plist in groups.values() for n, p in plist]
    per_comp = {}
    for name in COMPONENTS:
        for plist in all_groups.values():
            for _, p in plist:
                p.grad = None
        comps[name].backward(retain_graph=True)
        per_comp[name] = snap_grads(all_groups)
    for plist in all_groups.values():
        for _, p in plist:
            p.grad = None

    def cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    out = {"norms": {}, "cos": {}}
    for g in all_groups:
        out["norms"][g] = {c: float(per_comp[c][g].norm()) for c in COMPONENTS}
    for g in all_groups:
        for i, a in enumerate(COMPONENTS):
            for b in COMPONENTS[i + 1:]:
                out["cos"][f"{g}|cos(g_{a},g_{b})"] = cos(per_comp[a][g], per_comp[b][g])
    # weighted base = md + 1.5*(d+a) + cp (training composition, by linearity)
    W = {"maskdistill": 1.0, "rkd_sh_d": 1.5, "rkd_sh_a": 1.5, "couple": 1.0}
    for g in all_groups:
        g_base = sum(W[c] * per_comp[c][g] for c in W)
        out["norms"][g]["BASE"] = float(g_base.norm())
        for c in ("rel_rot", "rel_tdir"):
            out["cos"][f"{g}|cos(g_{c},g_base)"] = cos(per_comp[c][g], g_base)
    out["losses"] = {c: float(comps[c].detach()) for c in COMPONENTS}
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--base_root", required=True, help="rkdc1h run dir (no rel)")
    ap.add_argument("--rel_root", required=True, help="rkdc1hr run dir (with rel)")
    ap.add_argument("--check_a_pairs", nargs="+", type=int, default=[],
                    help="train-pair indices for the spike localization (e.g. 5 8 9)")
    ap.add_argument("--check_c_on", default="probe0",
                    help="input for Check C: probe0 | pair:<idx>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    fg.set_dataset(args.dataset)
    model_name = "model_weights/DA3-GIANT-1.1"

    teacher = P.create_teacher(model_name, "cpu")
    student = P.create_student(model_name, "cpu")

    all_results = {}
    for scene in args.scenes:
        print(f"\n=== {scene} ===", flush=True)
        data = fg.get_scene_data(scene)
        files = list(data.image_files)
        proto = P.build_scene_protocol(files, scene, dataset=args.dataset,
                                       n_train=10, n_shared=4, teacher_N=8)

        # ---- inputs: probes + requested train pairs ----
        input_specs = [(f"probe{k}", pair) for k, pair in
                       enumerate(proto["probe_pairs"][:2])]
        for idx in args.check_a_pairs:
            input_specs.append((f"pair{idx}", proto["train_pairs"][idx]))

        teacher.to(device)
        inputs = {}
        for name, pair in input_specs:
            imgs_t = torch.stack([load_images_da3([files[i]])[0]
                                  for i in pair["teacher_frames"]]).unsqueeze(0).to(device)
            ph, pw = imgs_t.shape[-2] // PATCH_SIZE, imgs_t.shape[-1] // PATCH_SIZE
            slots = [0, 2, 4, 6]
            inputs[name] = {
                "cache": P.cache_teacher_pair(teacher, imgs_t, slots, (ph, pw)),
                "imgs4": imgs_t[0, slots].cpu(),
                "frames": list(map(int, pair["teacher_frames"])),
                "slots": slots,
            }
            del imgs_t
        teacher.to("cpu")
        torch.cuda.empty_cache()
        patch_hw = inputs["probe0"]["cache"]["patch_hw"]
        head_norm = student.da3.model.head.norm

        def masked(name, k):
            """fixed, reproducible diagnostic masks (own namespace)"""
            imgs4 = inputs[name]["imgs4"].unsqueeze(0).to(device)
            gen = torch.Generator(device=device).manual_seed(
                P.stable_seed("diag2_mask", scene, name, k))
            imgs_m, _ = P.mask_image_blocks(imgs4, 0.5, patch_hw, gen)
            return imgs_m

        # ---- states ----
        student.to(device)
        torch.manual_seed(P.stable_seed("lora_init", scene, args.seed))
        P.reset_lora_(student)
        groups = classify_param_groups(student)
        n_trainable = sum(p.numel() for plist in groups.values() for _, p in plist)

        states = {}
        for label, root in [("theta_base", args.base_root), ("theta_rel", args.rel_root)]:
            ckpt = os.path.join(root, "ckpts", scene, "c2m_final_lora.pt")
            assert ckpt is not None and os.path.isfile(ckpt), \
                f"{label}: checkpoint missing (was silently ignored in v1): {ckpt}"
            assert os.path.isdir(ckpt.replace(".pt", "_peft")), ckpt
            states[label] = {"ckpt": ckpt, "sha256": sha256_of(ckpt)}

        scene_out = {"meta": {
            "dataset": args.dataset, "scene": scene, "seed": args.seed,
            "patch_hw": list(patch_hw),
            "n_trainable_params": n_trainable,
            "param_groups": {g: len(plist) for g, plist in groups.items()},
            "param_group_numels": {g: sum(p.numel() for _, p in plist)
                                   for g, plist in groups.items()},
            "inputs": {n: {"frames": v["frames"], "slots": v["slots"]}
                       for n, v in inputs.items()},
            "states": {"theta0": "seeded reset (training step-0 LoRA init)",
                       **{k: v for k, v in states.items()}},
        }, "states": {}}

        for label in ("theta0", "theta_base", "theta_rel"):
            if label == "theta0":
                torch.manual_seed(P.stable_seed("lora_init", scene, args.seed))
                P.reset_lora_(student)
            else:
                P.reset_lora_(student)
                student.load_lora_weights(states[label]["ckpt"])
            student.to(device).eval()

            st = {"inputs": {}, "grads": {}}
            # M1: components + per-pair decomposition
            for iname in inputs:
                conds = {}
                for cond in ["clean"] + [f"m{k}" for k in range(N_FIXED_MASKS)]:
                    imgs4 = (inputs[iname]["imgs4"].unsqueeze(0).to(device)
                             if cond == "clean" else masked(iname, int(cond[1:])))
                    with torch.no_grad():
                        comps_all = forward_all(student, inputs[iname]["cache"], imgs4,
                                                patch_hw, head_norm)
                        vals = {c: float(comps_all[c]) for c in COMPONENTS}
                        tap_feats, ext_s, depth_s = P.student_forward_c2m(
                            student, imgs4, with_depth=True)
                        rel_c, rows = decompose_pairs(
                            ext_s, inputs[iname]["cache"]["ext4"].to(device), depth_s)
                    vals.update(rel_c)
                    conds[cond] = {"components": vals, "pairs": rows}
                    del imgs4, comps_all
                st["inputs"][iname] = conds
                print(f"  [{label}] {iname} components+pairs done", flush=True)

            # M2: gradient matrix on probes, clean + m0
            for iname in ("probe0", "probe1"):
                if iname not in inputs:
                    continue
                for cond in ("clean", "m0"):
                    imgs4 = (inputs[iname]["imgs4"].unsqueeze(0).to(device)
                             if cond == "clean" else masked(iname, 0))
                    st["grads"][f"{iname}/{cond}"] = grad_matrix(
                        student, inputs[iname]["cache"], imgs4, patch_hw,
                        head_norm, groups)
                    del imgs4
                print(f"  [{label}] {iname} grad matrix done", flush=True)
            scene_out["states"][label] = st
            torch.cuda.empty_cache()

        # M3: Check C — counterfactual one-step AdamW updates
        if args.check_c_on.startswith("pair:"):
            cc_in = f"pair{args.check_c_on.split(':')[1]}"
        else:
            cc_in = "probe0"
        imgs_train = masked(cc_in, 0)
        cc_out = []
        for label in ("theta_base", "theta_rel"):
            P.reset_lora_(student)
            student.load_lora_weights(states[label]["ckpt"])
            student.to(device).eval()
            snapshot = {n: p.data.clone() for plist in groups.values()
                        for n, p in plist}
            params = [p for plist in groups.values() for _, p in plist]
            before = forward_all(student, inputs[cc_in]["cache"], imgs_train,
                                 patch_hw, head_norm)
            before_vals = {c: float(before[c].detach()) for c in COMPONENTS}
            probe_before = forward_all(
                student, inputs["probe0"]["cache"],
                inputs["probe0"]["imgs4"].unsqueeze(0).to(device),
                patch_hw, head_norm) if "probe0" in inputs else None
            branches = {}
            for bname, wts in [("base", {}),
                               ("base+R", {"rel_rot": 1.0}),
                               ("base+t", {"rel_tdir": 1.0}),
                               ("base+R+t", {"rel_rot": 1.0, "rel_tdir": 1.0})]:
                # restore theta, fresh optimizer (training hyperparams)
                for n, p in [(n, p) for plist in groups.values() for n, p in plist]:
                    p.data.copy_(snapshot[n])
                    p.grad = None
                opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=1e-5)
                comps = forward_all(student, inputs[cc_in]["cache"], imgs_train,
                                    patch_hw, head_norm)
                W = {"maskdistill": 1.0, "rkd_sh_d": 1.5, "rkd_sh_a": 1.5,
                     "couple": 1.0}
                total = sum(W[c] * comps[c] for c in W) + \
                    sum(w * comps[c] for c, w in wts.items())
                total.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                after = forward_all(student, inputs[cc_in]["cache"], imgs_train,
                                    patch_hw, head_norm)
                entry = {"grad_norm_preclip": float(gn),
                         "after_on_input": {c: float(after[c].detach()) for c in COMPONENTS}}
                if probe_before is not None:
                    pa = forward_all(student, inputs["probe0"]["cache"],
                                     inputs["probe0"]["imgs4"].unsqueeze(0).to(device),
                                     patch_hw, head_norm)
                    entry["after_on_probe0_clean"] = {c: float(pa[c].detach()) for c in COMPONENTS}
                branches[bname] = entry
            cc_out.append({"state": label, "input": f"{cc_in}/m0",
                           "before": before_vals,
                           "before_probe0_clean": {c: float(probe_before[c].detach())
                                                   for c in COMPONENTS} if probe_before else None,
                           "branches": branches})
            print(f"  [checkC] {label} done", flush=True)
            torch.cuda.empty_cache()
        scene_out["check_c"] = cc_out

        all_results[scene] = scene_out
        with open(args.out or os.path.join(
                ROOT, "workspace", f"fg_diag_v2_{args.dataset}.json"), "w") as f:
            json.dump(all_results, f, indent=1)
        del inputs
        torch.cuda.empty_cache()

    print("\nDONE. Results saved.")


if __name__ == "__main__":
    main()
