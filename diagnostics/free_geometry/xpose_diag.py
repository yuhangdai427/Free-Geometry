#!/usr/bin/env python3
"""Xpose failure diagnosis: why did the shared<->extra pose-anchor loss fail?

Per pair (6 scenes x 12, GT offline only), measure:
  a) Kabsch gauge-alignment residual (student 4 centers -> teacher), in units
     of mean inter-camera baseline (bigger = less trustworthy alignment).
  b) teacher shared->extra rel-pose error vs GT  (are extra anchors accurate?)
  c) student(aligned) shared->extra rel-pose error vs GT (is student worse?)
  -> gap = (c) - (b): the distillable signal. gap ~ 0 => the idea is dead by
  the gap-bounds-gain law, regardless of implementation.
Also record the xpose loss balance (xrel_sx vs xrel_ss magnitudes).

Output: artifacts/diagnostics/xpose_diag/summary.md
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, get_scene_data, load_manifest  # noqa: E402
import modeling as M  # noqa: E402
from train_arms import _kabsch_rt  # noqa: E402

EXTRA_INDICES = [i for i in range(8) if i not in STUDENT_INDICES]
OUT = "artifacts/diagnostics/xpose_diag"
HW = (378, 504)


@torch.no_grad()
def main():
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    os.makedirs(OUT, exist_ok=True)
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    rows = []
    for scene in sorted(manifest["scenes"]):
        sc = manifest["scenes"][scene]
        scene_data = get_scene_data(scene)
        M.reset_lora_(student); student.eval()
        for pi, pair in enumerate(sc["train_pairs"] + sc["probe_pairs"]):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"])
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            cache = M.cache_teacher_pair(teacher, images8, (ph, pw))
            _, _, sp = M.student_preds(student, images4)
            es, _ = pose_encoding_to_extri_intri(sp["pose_enc"].float(), HW)
            et, _ = pose_encoding_to_extri_intri(cache["pose_enc8"].float(), HW)
            Rs, ts = es[0, :, :3, :3], es[0, :, :3, 3]
            Rt, tt = et[0, :, :3, :3], et[0, :, :3, 3]
            gt = torch.from_numpy(np.asarray(scene_data.extrinsics)).float().cuda()
            g_sh = gt[pair["student_frames"]]
            g_ex = gt[[pair["teacher_frames"][i] for i in EXTRA_INDICES]]
            gR = torch.cat([g_sh[:, :3, :3], g_ex[:, :3, :3]])
            gt_ = torch.cat([g_sh[:, :3, 3], g_ex[:, :3, 3]])

            def angle_deg(R_):
                c = ((R_.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
                return c.acos().abs() * 180 / np.pi

            def rel_rot(R_i, R_j):
                return R_i.T @ R_j

            def rel_tdir(R_i, t_i, t_j):
                v = R_i.T @ (t_j - t_i)
                return v / v.norm().clamp_min(1e-8)

            def rel_set(R, t, idx_a, idx_b):
                Rs_, Ts_ = [], []
                for i in idx_a:
                    for j in idx_b:
                        Rs_.append(rel_rot(R[i], R[j]))
                        Ts_.append(rel_tdir(R[i], t[i], t[j]))
                return torch.stack(Rs_), torch.stack(Ts_)

            def rel_err_vs(R1, T1, R2, T2):
                ra = angle_deg(torch.stack([R1[k].T @ R2[k] for k in range(len(R1))])).mean().item()
                ta = (1 - (T1 * T2).sum(-1).clamp(-1, 1)).acos().abs().mean().item() * 180 / np.pi
                return ra, ta

            sh_idx, ex_idx = list(STUDENT_INDICES), list(EXTRA_INDICES)
            Rt_sh = rel_set(Rt, tt, sh_idx, ex_idx)
            GT_set = rel_set(gR, gt_, [0, 1, 2, 3], [4, 5, 6, 7])
            t_err = rel_err_vs(*Rt_sh, *GT_set)

            Ra, ta_ = _kabsch_rt(ts, tt[sh_idx])
            base_len = (tt[sh_idx][:, None, :] - tt[sh_idx][None, :, :]).norm(dim=-1).mean()
            resid = ((Ra @ ts.T).T + ta_ - tt[sh_idx]).norm(dim=-1).mean()
            Rs_al, ts_al = Ra @ Rs, (Ra @ ts.T).T + ta_
            Sx_set = rel_set(torch.cat([Rs_al, Rt[ex_idx]]), torch.cat([ts_al, tt[ex_idx]]),
                             [0, 1, 2, 3], [4, 5, 6, 7])
            s_err = rel_err_vs(*Sx_set, *GT_set)
            rows.append(dict(scene=scene, pair=pi,
                             align_resid=float(resid / base_len),
                             t_rot=t_err[0], t_t=t_err[1],
                             s_rot=s_err[0], s_t=s_err[1]))
            print(f"[{scene} p{pi}] resid {float(resid/base_len):.3f} "
                  f"T {t_err[0]:.1f}/{t_err[1]:.1f} S {s_err[0]:.1f}/{s_err[1]:.1f} "
                  f"gap {(s_err[0]-t_err[0]):+.1f}/{(s_err[1]-t_err[1]):+.1f}", flush=True)
        torch.cuda.empty_cache()
    import statistics as st
    lines = ["# xpose failure diagnosis (6 scenes x 12 pairs)", "",
             "| scene | pair | align_resid/baseline | T rot | T tdir | S rot | S tdir | gap rot | gap tdir |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['pair']} | {r['align_resid']:.3f} | {r['t_rot']:.1f} | {r['t_t']:.1f} | {r['s_rot']:.1f} | {r['s_t']:.1f} | {r['s_rot']-r['t_rot']:+.1f} | {r['s_t']-r['t_t']:+.1f} |")
    lines += ["", f"align_resid mean {st.mean(r['align_resid'] for r in rows):.3f}",
              f"T rel err (deg): rot {st.mean(r['t_rot'] for r in rows):.1f} tdir {st.mean(r['t_t'] for r in rows):.1f}",
              f"S rel err (deg): rot {st.mean(r['s_rot'] for r in rows):.1f} tdir {st.mean(r['s_t'] for r in rows):.1f}",
              f"GAP (S-T): rot {st.mean(r['s_rot']-r['t_rot'] for r in rows):+.1f} tdir {st.mean(r['s_t']-r['t_t'] for r in rows):+.1f}",
              "", "gap~0 => nothing to distill (dead); gap>0 & resid small => fixable arm"]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
