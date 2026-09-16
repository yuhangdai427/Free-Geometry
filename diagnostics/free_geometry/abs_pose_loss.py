"""SelfEvo-style absolute pseudo-label camera loss with gauge normalization.

Teacher (frozen base model, full-context forward) poses are decoded, normalized
(first shared frame as coordinate origin + mean world-point distance as unit
scale, following normalize_camera_extrinsics_and_points_batch in
vggt/training/train_utils/normalization.py and SelfEvo trainer.py:1934-1951),
and re-encoded as pseudo-GT pose_enc targets. The student's raw poses are pulled
to them with component-wise L1 (T/R/FL = 1/1/0.5, translation error clamped at
100), mirroring SelfEvo's compute_camera_loss (loss.py:81-197 in SelfEvo).
"""

import numpy as np
import torch

from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from common import STUDENT_INDICES

IMAGE_HW = (378, 504)


def _to_homog(E):
    # [B,N,3,4] -> [B,N,4,4]
    B, N, _, _ = E.shape
    H = torch.eye(4, device=E.device, dtype=E.dtype).reshape(1, 1, 4, 4).expand(B, N, 4, 4).clone()
    H[:, :, :3, :4] = E
    return H


@torch.no_grad()
def _teacher_target(pose_enc_t, depth4, conf4, shared_idx, image_hw=IMAGE_HW):
    """Normalized pseudo-GT pose_enc for the shared frames + the scale used.

    pose_enc_t: [1,N,9] teacher pose encodings (all teacher frames)
    depth4:     [1,S,H,W,1] teacher depths on shared frames (student order)
    conf4:      [1,S,H,W,1] teacher confidence on shared frames (or None)
    shared_idx: teacher slot index of each shared frame (e.g. [0,2,4,6])
    """
    E_t, K_t = pose_encoding_to_extri_intri(pose_enc_t.float(), image_hw)  # w2c [1,N,3,4]
    H_t = _to_homog(E_t)
    E0_inv = closed_form_inverse_se3(H_t[:, 0])          # [1,4,4]
    E_norm = (H_t @ E0_inv.unsqueeze(1))[:, :, :3, :4]   # poses in frame-0 coords

    # scale: mean distance of teacher points from the first camera
    d = depth4.squeeze(0).squeeze(-1).float()            # [S,H,W]
    c = conf4.squeeze(0).squeeze(-1).float() if conf4 is not None else None
    S, H, W = d.shape
    ys, xs = torch.meshgrid(torch.arange(0, H, 6, device=d.device),
                            torch.arange(0, W, 6, device=d.device), indexing="ij")
    dists = []
    for k in range(S):
        dk = d[k][ys, xs]
        valid = torch.isfinite(dk) & (dk > 0)
        if c is not None and valid.any():
            ck = c[k][ys, xs]
            q = torch.quantile(ck[valid].flatten(), 0.05)
            valid = valid & (ck >= q)
        if int(valid.sum()) < 100:
            continue
        slot = int(shared_idx[k])
        Kk = K_t[0, slot].float()
        Z = dk[valid]
        X = (xs[valid].float() + 0.5 - Kk[0, 2]) * Z / Kk[0, 0]
        Y = (ys[valid].float() + 0.5 - Kk[1, 2]) * Z / Kk[1, 1]
        pts_cam = torch.stack([X, Y, Z], dim=-1)
        E_norm_k = _to_homog(E_norm)[:, slot]                       # cam_0 -> cam_k
        w0_from_cam_k = closed_form_inverse_se3(E_norm_k)           # cam_k -> cam_0
        pts_h = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=-1)  # [P,4]
        pts_c0 = (w0_from_cam_k @ pts_h.mT).mT[0, :, :3]                       # [P,3]
        dists.append(pts_c0.norm(dim=-1))
    if not dists:
        avg_scale = torch.ones((), device=E_t.device)
    else:
        avg_scale = torch.cat(dists).mean().clamp(min=1e-6, max=1e6)

    E_scaled = E_norm.clone()
    E_scaled[:, :, :3, 3] = E_scaled[:, :, :3, 3] / avg_scale
    shared = list(shared_idx)
    target = extri_intri_to_pose_encoding(E_scaled[:, shared], K_t[:, shared],
                                          image_hw, pose_encoding_type="absT_quaR_FoV")
    return target.detach(), avg_scale.detach()


def loss_abs_pose_norm(pose_enc_s, teacher_cache, image_hw=IMAGE_HW,
                       w_t=1.0, w_r=1.0, w_fl=0.5):
    """L1 on pose_enc components against gauge-normalized teacher pseudo-GT."""
    with torch.autocast(device_type="cuda", enabled=False):
        ps = pose_enc_s.float()
        tgt, scale = _teacher_target(teacher_cache["pose_enc8"].float(),
                                     teacher_cache["depth4"],
                                     teacher_cache.get("conf4"),
                                     STUDENT_INDICES, image_hw)
        # sign-match quaternions (avoid +/-q ambiguity)
        qs, qt = ps[..., 3:7], tgt[..., 3:7]
        flip = torch.where((qs * qt).sum(-1, keepdim=True) < 0,
                           -torch.ones_like(qs), torch.ones_like(qs))
        qs = qs * flip
        lt = (ps[..., :3] - tgt[..., :3]).abs().clamp(max=100.0).mean()
        lr = (qs - qt).abs().mean()
        lfl = (ps[..., 7:] - tgt[..., 7:]).abs().mean()
        loss = w_t * lt + w_r * lr + w_fl * lfl
    return loss, {"abs_T": float(lt), "abs_R": float(lr), "abs_FL": float(lfl),
                  "abs_scale": float(scale)}


def loss_abs_pose_raw(pose_enc_s, pose_enc_t_shared, image_hw=IMAGE_HW,
                      beta=0.1, w_fl=0.5):
    """Absolute pose distillation against the teacher's RAW pose_enc (no gauge
    normalization). Both sides are first-frame anchored by architecture.
    Rotation: chordal on decoded 3x3 matrices (no quaternion pitfalls).
    Translation: Huber(beta) on raw absolute translation.
    FoV: L1 on pose_enc dims 7:9."""
    with torch.autocast(device_type="cuda", enabled=False):
        ps = pose_enc_s.float()
        pt = pose_enc_t_shared.float()
        E_s, _ = pose_encoding_to_extri_intri(ps, image_hw)
        E_t, _ = pose_encoding_to_extri_intri(pt, image_hw)
        Rs, ts = E_s[..., :3, :3], E_s[..., :3, 3]
        Rt, tt = E_t[..., :3, :3], E_t[..., :3, 3]
        r = ((Rs - Rt) ** 2).sum(dim=(-2, -1)).clamp_min(1e-12).sqrt().mean()
        t_l = torch.nn.functional.huber_loss(ts, tt, delta=beta)
        fl = (ps[..., 7:] - pt[..., 7:]).abs().mean()
    return r + t_l + w_fl * fl, {"absr_R": float(r), "absr_T": float(t_l),
                                 "absr_FL": float(fl)}


def loss_abs_t_fl(pose_enc_s, pose_enc_t_shared, image_hw=IMAGE_HW,
                  beta=0.1, w_fl=0.5):
    """Absolute translation (Huber) + FoV (L1) anchoring to teacher RAW poses.
    Rotation deliberately excluded: it is covered by rel-pose (relative form)."""
    with torch.autocast(device_type="cuda", enabled=False):
        ps = pose_enc_s.float()
        pt = pose_enc_t_shared.float()
        E_s, _ = pose_encoding_to_extri_intri(ps, image_hw)
        E_t, _ = pose_encoding_to_extri_intri(pt, image_hw)
        lt = torch.nn.functional.huber_loss(E_s[..., :3, 3], E_t[..., :3, 3], delta=beta)
        lfl = (ps[..., 7:] - pt[..., 7:]).abs().mean()
    return lt + w_fl * lfl, {"absT": float(lt), "absFL": float(lfl)}


def loss_rkd_triplet_pose(pose_enc_s5, pose_enc_t, shared_slots_t, anchor_s,
                          anchor_t, image_hw=IMAGE_HW):
    """RKD-style triplet geometry in pose space, gauge-free.

    Anchor frame N (present on both sides). For every unordered shared pair
    (i,j): match the triangle shape through N —
      angle at N between translation directions N->i and N->j  (cos value)
      log edge-length ratio log(|N->i| / |N->j|)
    Both quantities are invariant to global rotation/translation/scale.

    pose_enc_s5: [1,5,9] student poses (frames 0..3 shared, frame anchor_s = N)
    pose_enc_t:  [1,N,9] teacher poses; shared frames at shared_slots_t, N at anchor_t
    """
    from common import STUDENT_INDICES  # unused here; kept for symmetry

    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)

        cs = centers(pose_enc_s5)[0]                      # [5,3]
        ct_all = centers(pose_enc_t)[0]                   # [N,3]
        ct = ct_all[list(shared_slots_t)]                 # [4,3]
        cN_s, cN_t = cs[anchor_s], ct_all[anchor_t]
        terms = []
        for i in range(4):
            for j in range(i + 1, 4):
                vi_s = torch.nn.functional.normalize(cs[i] - cN_s, dim=-1, eps=1e-8)
                vj_s = torch.nn.functional.normalize(cs[j] - cN_s, dim=-1, eps=1e-8)
                vi_t = torch.nn.functional.normalize(ct[i] - cN_t, dim=-1, eps=1e-8)
                vj_t = torch.nn.functional.normalize(ct[j] - cN_t, dim=-1, eps=1e-8)
                ang_s = (vi_s * vj_s).sum()
                ang_t = (vi_t * vj_t).sum()
                d_i_s = (cs[i] - cN_s).norm().clamp_min(1e-6)
                d_j_s = (cs[j] - cN_s).norm().clamp_min(1e-6)
                d_i_t = (ct[i] - cN_t).norm().clamp_min(1e-6)
                d_j_t = (ct[j] - cN_t).norm().clamp_min(1e-6)
                lr_s = torch.log(d_i_s / d_j_s)
                lr_t = torch.log(d_i_t / d_j_t)
                terms.append((ang_s - ang_t) ** 2 + (lr_s - lr_t) ** 2)
        loss = torch.stack(terms).mean()
    return loss, {"rkd_tri": float(loss)}


def loss_rkd_shared_pose(pose_enc_s, pose_enc_t_shared, image_hw=IMAGE_HW):
    """Anchor-free RKD on the 4 shared frames' camera centers (gauge-free).
    Shape match: mean-normalized 6 pairwise distances + 12 triangle angles.
    Complements rel-pose (which drops translation norms) by supervising the
    trajectory's shape; no anchor frame, no extra forward."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        cs, ct = centers(pose_enc_s), centers(pose_enc_t_shared)
        S = cs.shape[0]
        pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
        ds = torch.stack([(cs[i] - cs[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dt = torch.stack([(ct[i] - ct[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dn_s = ds / ds.mean().clamp_min(1e-8)
        dn_t = dt / dt.mean().clamp_min(1e-8)
        dloss = ((dn_s - dn_t) ** 2).mean()
        aterms = []
        for k in range(S):
            rest = [x for x in range(S) if x != k]
            for a, b in [(rest[0], rest[1]), (rest[0], rest[2]), (rest[1], rest[2])]:
                vs1 = torch.nn.functional.normalize(cs[a] - cs[k], dim=-1, eps=1e-8)
                vs2 = torch.nn.functional.normalize(cs[b] - cs[k], dim=-1, eps=1e-8)
                vt1 = torch.nn.functional.normalize(ct[a] - ct[k], dim=-1, eps=1e-8)
                vt2 = torch.nn.functional.normalize(ct[b] - ct[k], dim=-1, eps=1e-8)
                aterms.append(((vs1 * vs2).sum() - (vt1 * vt2).sum()) ** 2)
        aloss = torch.stack(aterms).mean()
        loss = dloss + aloss
    return loss, {"rkd_sh_d": float(dloss), "rkd_sh_a": float(aloss)}


def loss_rkd_shared_pose_huber(pose_enc_s, pose_enc_t_shared, delta=0.2, image_hw=IMAGE_HW):
    """Huberized variant of loss_rkd_shared_pose: per-residual Huber (delta)
    instead of plain squares, so scenes with a large student-teacher shape gap
    get linear (bounded) gradients instead of quadratic blow-up."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        cs, ct = centers(pose_enc_s), centers(pose_enc_t_shared)
        S = cs.shape[0]
        pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
        ds = torch.stack([(cs[i] - cs[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dt = torch.stack([(ct[i] - ct[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dn_s = ds / ds.mean().clamp_min(1e-8)
        dn_t = dt / dt.mean().clamp_min(1e-8)
        dloss = torch.nn.functional.huber_loss(dn_s, dn_t, delta=delta)
        aterms = []
        for k in range(S):
            rest = [x for x in range(S) if x != k]
            for a, b in [(rest[0], rest[1]), (rest[0], rest[2]), (rest[1], rest[2])]:
                vs1 = torch.nn.functional.normalize(cs[a] - cs[k], dim=-1, eps=1e-8)
                vs2 = torch.nn.functional.normalize(cs[b] - cs[k], dim=-1, eps=1e-8)
                vt1 = torch.nn.functional.normalize(ct[a] - ct[k], dim=-1, eps=1e-8)
                vt2 = torch.nn.functional.normalize(ct[b] - ct[k], dim=-1, eps=1e-8)
                aterms.append(torch.nn.functional.huber_loss(
                    (vs1 * vs2).sum(), (vt1 * vt2).sum(), delta=delta))
        aloss = torch.stack(aterms).mean()
        loss = dloss + aloss
    return loss, {"rkd_sh_d": float(dloss), "rkd_sh_a": float(aloss)}
def loss_rkd_local_huber(pose_enc_s, pose_enc_t_shared, delta=0.2, window=2, image_hw=IMAGE_HW):
    """Local-window RKD: shape invariants only over anchor pairs |i-j|<=window
    (in shared-frame order) + consecutive-triple angles. Avoids forcing the
    student to match the teacher's LONG-RANGE shape (unreliable over long
    spans); keeps local geometry supervision. Huber-bounded residuals."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        cs, ct = centers(pose_enc_s), centers(pose_enc_t_shared)
        S = cs.shape[0]
        pairs = [(i, j) for i in range(S) for j in range(i + 1, S) if j - i <= window]
        ds = torch.stack([(cs[i] - cs[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dt = torch.stack([(ct[i] - ct[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dn_s = ds / ds.mean().clamp_min(1e-8)
        dn_t = dt / dt.mean().clamp_min(1e-8)
        dloss = torch.nn.functional.huber_loss(dn_s, dn_t, delta=delta)
        aterms = []
        for k in range(S - 2):
            a, b = k + 1, k + 2
            vs1 = torch.nn.functional.normalize(cs[a] - cs[k], dim=-1, eps=1e-8)
            vs2 = torch.nn.functional.normalize(cs[b] - cs[k], dim=-1, eps=1e-8)
            vt1 = torch.nn.functional.normalize(ct[a] - ct[k], dim=-1, eps=1e-8)
            vt2 = torch.nn.functional.normalize(ct[b] - ct[k], dim=-1, eps=1e-8)
            aterms.append(torch.nn.functional.huber_loss(
                (vs1 * vs2).sum(), (vt1 * vt2).sum(), delta=delta))
        aloss = torch.stack(aterms).mean() if aterms else torch.zeros((), device=cs.device)
        loss = dloss + aloss
    return loss, {"rkd_loc_d": float(dloss), "rkd_loc_a": float(aloss)}


def loss_rkd_mix_pose(pose_enc_s4, pose_enc_t, shared_slots_t, anchor_t,
                      image_hw=IMAGE_HW):
    """Mixed-gauge RKD: student's shared-frame centers against the teacher's
    anchor N taken AS-IS (teacher gauge). NOT gauge-free by design: the mixed
    triangle's angle/ratio implicitly pulls the student's gauge toward the
    teacher's. Target side = teacher's own triangle (same gauge throughout).
    """
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        cs = centers(pose_enc_s4)                        # [4,3] student gauge
        ct_all = centers(pose_enc_t)                     # [N,3] teacher gauge
        ct = ct_all[list(shared_slots_t)]
        cN_t = ct_all[anchor_t]                          # anchor in teacher gauge
        terms = []
        for i in range(4):
            for j in range(i + 1, 4):
                vm_i = torch.nn.functional.normalize(cs[i] - cN_t, dim=-1, eps=1e-8)
                vm_j = torch.nn.functional.normalize(cs[j] - cN_t, dim=-1, eps=1e-8)
                vt_i = torch.nn.functional.normalize(ct[i] - cN_t, dim=-1, eps=1e-8)
                vt_j = torch.nn.functional.normalize(ct[j] - cN_t, dim=-1, eps=1e-8)
                ang = ((vm_i * vm_j).sum() - (vt_i * vt_j).sum()) ** 2
                dm_i = (cs[i] - cN_t).norm().clamp_min(1e-6)
                dm_j = (cs[j] - cN_t).norm().clamp_min(1e-6)
                dt_i = (ct[i] - cN_t).norm().clamp_min(1e-6)
                dt_j = (ct[j] - cN_t).norm().clamp_min(1e-6)
                lr = (torch.log(dm_i / dm_j) - torch.log(dt_i / dt_j)) ** 2
                terms.append(ang + lr)
        loss = torch.stack(terms).mean()
    return loss, {"rkd_mix": float(loss)}


def loss_xac_camtok(camera_head, teacher_cache, feats24_s, tau=0.1):
    """Cross-frame camera-token affinity KD (feature space, no extra forward).

    For each shared frame i: the cosine-affinity distribution of the student's
    camera token over the teacher's EXTRA-frame camera tokens (the scene-context
    landmarks) must match the teacher's own distribution. KL(teacher || student).
    Post-LN camera tokens (camera_head.token_norm), layer 23.
    """
    from common import STUDENT_INDICES as SI
    with torch.autocast(device_type="cuda", enabled=False):
        N = teacher_cache["pose_enc8"].shape[1]
        extras = [i for i in range(N) if i not in SI]
        ctok_t_all = camera_head.token_norm(
            teacher_cache["feats"][23][0, :, 0, :].float()).detach()   # [N,C]
        K = ctok_t_all[extras]                                         # [E,C]
        ctok_t_shared = ctok_t_all[list(SI)]                           # [4,C]
        ctok_s = camera_head.token_norm(feats24_s[23][0, :, 0, :].float())  # [4,C]
        K = torch.nn.functional.normalize(K, dim=-1, eps=1e-8)
        qs = torch.nn.functional.normalize(ctok_s, dim=-1, eps=1e-8)
        qt = torch.nn.functional.normalize(ctok_t_shared, dim=-1, eps=1e-8)
        ps = torch.softmax((qs @ K.T) / tau, dim=-1)                   # [4,E]
        pt = torch.softmax((qt @ K.T) / tau, dim=-1).detach()
        loss = torch.nn.functional.kl_div(ps.log(), pt, reduction="batchmean")
    return loss, {"xac": float(loss)}


def loss_xac2_camtok(camera_head, teacher_cache, feats24_s, tau=0.1):
    """Flipped cross-frame camera-token affinity KD: teacher's EXTRA-frame tokens
    are the queries; the shared frames' camera tokens are the keys, compared
    between teacher (K_t) and student (K_s) versions. 'The scene looking at the
    shared frames must see the same interface.' KL(p_teacher || p_student).
    """
    from common import STUDENT_INDICES as SI
    with torch.autocast(device_type="cuda", enabled=False):
        N = teacher_cache["pose_enc8"].shape[1]
        extras = [i for i in range(N) if i not in SI]
        ctok_t_all = camera_head.token_norm(
            teacher_cache["feats"][23][0, :, 0, :].float()).detach()   # [N,C]
        Q = torch.nn.functional.normalize(ctok_t_all[extras], dim=-1, eps=1e-8)  # [E,C]
        Kt = torch.nn.functional.normalize(ctok_t_all[list(SI)], dim=-1, eps=1e-8)  # [4,C]
        Ks = torch.nn.functional.normalize(
            camera_head.token_norm(feats24_s[23][0, :, 0, :].float()), dim=-1, eps=1e-8)
        pt = torch.softmax((Q @ Kt.T) / tau, dim=-1)                   # [E,4]
        ps = torch.softmax((Q @ Ks.T) / tau, dim=-1)
        loss = torch.nn.functional.kl_div(ps.log(), pt, reduction="batchmean")
    return loss, {"xac2": float(loss)}


def loss_scale_gauge(pose_enc_s, pose_enc_t_shared, image_hw=IMAGE_HW):
    """Single-scalar global-scale gauge anchor: (log s_s - log s_t)^2 where
    s = RMS spread of the 4 shared camera centers. One scalar per forward,
    no per-frame forces — cannot cause leveling. Targets the train-time side
    of the gauge-decoupling mechanism (translation-gauge drift)."""
    with torch.autocast(device_type="cuda", enabled=False):
        def spread(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            c = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
            return (c - c.mean(dim=1, keepdim=True)).norm(dim=-1).pow(2).mean(dim=1).sqrt()

        s_s = spread(pose_enc_s)                      # [1]
        s_t = spread(pose_enc_t_shared).detach()
        loss = (torch.log(s_s.clamp_min(1e-6)) - torch.log(s_t.clamp_min(1e-6))) ** 2
    return loss.squeeze(), {"scale_g": float(loss), "scale_ratio": float(s_s / s_t)}


def loss_couple(pose_enc_s, depth_s, pose_enc_t_shared, depth4_t, conf4_t=None,
                image_hw=IMAGE_HW):
    """Cross-head gauge coupling scalar: (log(RMS_centers / mean_depth)) aligned
    student vs teacher (teacher-shared 4 frames). One scalar per forward — the
    train-time proxy of the s_pose/s_depth decoupling mechanism."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        def couple_stat(pose, depth, conf=None):
            c = centers(pose)
            spr = (c - c.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()
            d = depth.squeeze(0).squeeze(-1).float() if depth.dim() >= 4 else depth.float()
            m = torch.isfinite(d) & (d > 0)
            if conf is not None:
                cf = conf.squeeze(0).squeeze(-1).float() if conf.dim() >= 4 else conf.float()
                q = torch.quantile(cf[m].flatten(), 0.05)
                m = m & (cf >= q)
            md = d[m].mean().clamp_min(1e-6)
            return torch.log(spr.clamp_min(1e-6)) - torch.log(md)

        cs = couple_stat(pose_enc_s, depth_s)
        ct = couple_stat(pose_enc_t_shared, depth4_t, conf4_t).detach()
        loss = (cs - ct) ** 2
    return loss.squeeze(), {"couple": float(loss)}


_RKD_EXT_TRIPLES = None


def loss_rkd_ext_pose(pose_enc_s, pose_enc_t, shared_slots_t, image_hw=IMAGE_HW,
                      n_ang=256):
    """Extended-set RKD: same invariants as loss_rkd_shared_pose but over the
    16-point cloud = 4 shared (student's own) + 12 extras (teacher's, identity
    assumption: the student is treated as sharing the teacher's extra-frame
    centers). Teacher side = all-16 teacher centers. Gauge-free invariants
    (mean-normalized pairwise distances + triangle angles, subsampled)."""
    global _RKD_EXT_TRIPLES
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        cs4 = centers(pose_enc_s)                    # [4,3] student
        ct16 = centers(pose_enc_t).detach()          # [16,3] teacher
        extras = [i for i in range(ct16.shape[0]) if i not in shared_slots_t]
        cS = torch.cat([cs4, ct16[extras]], dim=0)   # [16,3] student-side cloud
        cT = ct16

        def invariants(c):
            D = torch.cdist(c, c)                    # [16,16]
            iu = torch.triu_indices(16, 16, offset=1, device=c.device)
            d = D[iu[0], iu[1]].clamp_min(1e-6)
            dn = d / d.mean().clamp_min(1e-8)
            return dn

        dn_s, dn_t = invariants(cS), invariants(cT)
        dloss = ((dn_s - dn_t) ** 2).mean()

        if _RKD_EXT_TRIPLES is None:
            rng = np.random.RandomState(0)
            all_tr = [(k, a, b) for k in range(16) for a in range(16) for b in range(16)
                      if len({k, a, b}) == 3 and a < b]
            idx = rng.choice(len(all_tr), size=min(n_ang, len(all_tr)), replace=False)
            _RKD_EXT_TRIPLES = [all_tr[i] for i in idx]

        def ang_at(c, k, a, b):
            v1 = torch.nn.functional.normalize(c[a] - c[k], dim=-1, eps=1e-8)
            v2 = torch.nn.functional.normalize(c[b] - c[k], dim=-1, eps=1e-8)
            return (v1 * v2).sum()

        aloss = torch.stack([(ang_at(cS, k, a, b) - ang_at(cT, k, a, b)) ** 2
                             for k, a, b in _RKD_EXT_TRIPLES]).mean()
        loss = dloss + aloss
    return loss, {"rkd_ext_d": float(dloss), "rkd_ext_a": float(aloss)}


def loss_couple16(pose_enc_s, depth_s, pose_enc_t16, depth4_t, conf4_t=None,
                  image_hw=IMAGE_HW):
    """Couple with teacher spread over ALL 16 frames (pull 4v gauge toward the
    16v regime), depth mean on the shared 4 frames both sides (same-frame
    comparable)."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        def spread(c):
            return (c - c.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()

        def dmean(depth, conf=None):
            d = depth.squeeze(0).squeeze(-1).float() if depth.dim() >= 4 else depth.float()
            m = torch.isfinite(d) & (d > 0)
            if conf is not None:
                cf = conf.squeeze(0).squeeze(-1).float() if conf.dim() >= 4 else conf.float()
                q = torch.quantile(cf[m].flatten(), 0.05)
                m = m & (cf >= q)
            return d[m].mean().clamp_min(1e-6)

        cs = torch.log(spread(centers(pose_enc_s)).clamp_min(1e-6)) - torch.log(dmean(depth_s))
        ct = (torch.log(spread(centers(pose_enc_t16)).clamp_min(1e-6))
              - torch.log(dmean(depth4_t, conf4_t))).detach()
        loss = (cs - ct) ** 2
    return loss.squeeze(), {"couple16": float(loss)}


def loss_xap_pool(depth_head, teacher_cache, feats24_s, tau=0.1):
    """Pooled-patch frame descriptor affinity KD. Per-frame token = mean of patch
    tokens (layer 23, post DPT-LN via M.to_norm). Queries = teacher extra-frame
    pooled tokens; keys = shared-frame pooled tokens (teacher vs student).
    KL(p_teacher || p_student). Isolates the token-OBJECT variable vs XAC2."""
    from common import STUDENT_INDICES as SI
    import modeling as M
    with torch.autocast(device_type="cuda", enabled=False):
        def pool(feats):
            return M.to_norm(depth_head, M.to_patch(feats.float())).mean(dim=2)[0]  # [S,C]

        pt_all = pool(teacher_cache["feats"][23]).detach()             # [N,C]
        ps4 = pool(feats24_s[23])                                      # [4,C]
        N = pt_all.shape[0]
        extras = [i for i in range(N) if i not in SI]
        Q = torch.nn.functional.normalize(pt_all[extras], dim=-1, eps=1e-8)      # [E,C]
        Kt = torch.nn.functional.normalize(pt_all[list(SI)], dim=-1, eps=1e-8)   # [4,C]
        Ks = torch.nn.functional.normalize(ps4, dim=-1, eps=1e-8)
        pt = torch.softmax((Q @ Kt.T) / tau, dim=-1)
        ps = torch.softmax((Q @ Ks.T) / tau, dim=-1)
        loss = torch.nn.functional.kl_div(ps.log(), pt, reduction="batchmean")
    return loss, {"xap": float(loss)}


def loss_closure_align(pose_enc_s, pose_enc_t_shared, image_hw=IMAGE_HW):
    """Closure-residual alignment on the 4 shared frames (drift proxy).
    For chains i->k->j vs direct i->j: residual = chordal(R_chain, R_dir) +
    (1 - cos(t_chain, t_dir)). Student's residual matched to teacher's
    (residual ALIGNMENT, not zeroing — the CYC lesson)."""
    with torch.autocast(device_type="cuda", enabled=False):
        def Rt(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            return E[0, :, :3, :3], E[0, :, :3, 3]

        Rs, ts = Rt(pose_enc_s)
        Rt_, tt = Rt(pose_enc_t_shared)

        def relp(R, t, i, j):
            Rr = R[i].transpose(-1, -2) @ R[j]
            tr = (R[i].transpose(-1, -2) @ (t[j] - t[i]).unsqueeze(-1)).squeeze(-1)
            return Rr, tr

        def resid(R, t, chain):
            # chain = list of hops, e.g. [0,1,2]; compare composed with direct first->last
            Rc = None
            tc = None
            for a, b in zip(chain[:-1], chain[1:]):
                Rr, tr = relp(R, t, a, b)
                if Rc is None:
                    Rc, tc = Rr, tr
                else:
                    tc = tc + (Rc @ tr.unsqueeze(-1)).squeeze(-1)
                    Rc = Rc @ Rr
            Rd, td = relp(R, t, chain[0], chain[-1])
            r = (Rc - Rd).pow(2).sum().clamp_min(1e-12).sqrt()
            tn_c = torch.nn.functional.normalize(tc, dim=-1, eps=1e-8)
            tn_d = torch.nn.functional.normalize(td, dim=-1, eps=1e-8)
            return r + (1.0 - (tn_c * tn_d).sum())

        chains = [(0, 1, 2), (1, 2, 3), (0, 1, 2, 3)]
        rs = torch.stack([resid(Rs, ts, c) for c in chains])
        rt = torch.stack([resid(Rt_, tt, c) for c in chains]).detach()
        loss = ((rs - rt) ** 2).mean()
    return loss, {"clo": float(loss), "clo_s": float(rs.mean()), "clo_t": float(rt.mean())}


def loss_gram_pool(depth_head, teacher_cache, feats24_s):
    """Gram-matrix matching on per-frame pooled patch tokens (token-space RKD).
    Off-diagonal entries of the 4x4 cosine-similarity Gram matrix, student vs
    teacher (shared frames, post-LN pooled tokens)."""
    from common import STUDENT_INDICES as SI
    import modeling as M
    with torch.autocast(device_type="cuda", enabled=False):
        def pool(feats):
            return M.to_norm(depth_head, M.to_patch(feats.float())).mean(dim=2)[0]

        pt = torch.nn.functional.normalize(pool(teacher_cache["feats"][23][:, SI]).detach(), dim=-1, eps=1e-8)
        ps = torch.nn.functional.normalize(pool(feats24_s[23]), dim=-1, eps=1e-8)
        Gt = pt @ pt.T
        Gs = ps @ ps.T
        iu = torch.triu_indices(4, 4, offset=1, device=ps.device)
        loss = ((Gs[iu[0], iu[1]] - Gt[iu[0], iu[1]]) ** 2).mean()
    return loss, {"gram": float(loss)}
