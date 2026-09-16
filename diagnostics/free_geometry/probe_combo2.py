#!/usr/bin/env python3
"""Combo headroom probe: teacher:student view-count combinations.

For each (dataset, scene, combo, pair): teacher forward on T frames, student
forward on S frames (S = equidistant subset of T); teacher evaluated on the
shared views via frozen head replay. Metrics: pose AUC@3 T vs S, AbsRel T vs S,
frac patches T better. Selection follows the v1.1 branch (dense equidistant
spread for tau>0.55, else random); S = sorted(T)[::len(T)//n_s][:n_s].

Run from repo root:
  python diagnostics/free_geometry/probe_combo.py
"""

import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common  # noqa: E402
from common import get_scene_data, stable_seed  # noqa: E402
import modeling as M  # noqa: E402
from headroom_probe import load_gt_any  # noqa: E402
from depth_metrics import frame_metrics  # noqa: E402
from train_arms import perpatch_logres2  # noqa: E402

OUT = os.path.join(_REPO := os.path.abspath(os.path.join(_HERE, "..", "..")),
                   "artifacts", "diagnostics", "combo_probe2")

SCENES = {
    "scannetpp": ["7831862f02", "bde1e479ad", "09c1414f1b"],
    "7scenes": ["chess", "office", "stairs"],
    "hiroom": ["20241230/828738/cam_sampled_08", "20241230/828749/cam_sampled_12",
               "20241230/828766/cam_sampled_10"],
    "eth3d": ["courtyard", "office", "relief"],
}
TAU = {"scannetpp": 0.45, "7scenes": 0.70, "hiroom": 0.20, "eth3d": 0.13}
COMBOS = [(16, 4), (32, 4)]
N_PAIRS = 4


def build_pair(N, tN, n_s, dense, r):
    if dense:
        step = N / tN
        off = r.uniform(0, step)
        T = sorted(set(int(off + i * step) for i in range(tN)))[:tN]
    else:
        T = sorted(r.sample(range(N), tN))
    S = sorted(T[:: tN // n_s][:n_s])
    return T, S


def combo_metrics(teacher, student, base, scene_data, T, S, orig_hw):
    imgsT = np.stack([common.load_image_model(scene_data.image_files[i]) for i in T])
    imgsS = np.stack([common.load_image_model(scene_data.image_files[i]) for i in S])
    tT = torch.from_numpy(imgsT).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
    tS = torch.from_numpy(imgsS).permute(0, 3, 1, 2).float().unsqueeze(0).cuda()
    with torch.no_grad():
        tfeats, tpsi = M.aggregator_all(teacher, tT)
        sfeats, spsi = student._get_aggregator()(tS)
    slots = [None] * 24
    idx = [T.index(f) for f in S]
    for l in [4, 11, 17, 23]:
        slots[l] = tfeats[l][:, idx].float().contiguous()
    with torch.no_grad():
        dt, _ = M.replay_depth_nograd(base, slots, tS, tpsi)
        ds_, _ = M.replay_depth_nograd(base, [f.float() for f in sfeats], tS, spsi)
        pose_t = M.replay_camera_nograd(base, slots)
        pose_s = M.replay_camera_nograd(base, [f.float() for f in sfeats])
    dt = dt.squeeze(0).squeeze(-1).float().cpu().numpy()
    ds_ = ds_.squeeze(0).squeeze(-1).float().cpu().numpy()
    gt_ext = np.asarray(scene_data.extrinsics)[S]
    auc_t = M.pose_auc(pose_t.float(), gt_ext)["auc03"]
    auc_s = M.pose_auc(pose_s.float(), gt_ext)["auc03"]
    H, W = dt.shape[-2:]
    gt = np.stack([load_gt_any(scene_data, f, (H, W), orig_hw) for f in S], 0)
    valid = np.isfinite(gt) & (gt > 0)
    if valid.sum() < 1000:
        return None

    def scaled_absrel(d):
        r = np.log(np.clip(d, 1e-6, None)) - np.log(np.clip(gt, 1e-6, None))
        c = r[valid].mean()
        d2 = d * np.exp(-c)
        vals = [frame_metrics(d2[k], gt[k])["absrel"] for k in range(len(S))
                if frame_metrics(d2[k], gt[k])]
        return float(np.mean(vals))

    ar_t, ar_s = scaled_absrel(dt), scaled_absrel(ds_)
    ph, pw = H // 14, W // 14
    e_t = perpatch_logres2(dt, gt, (ph, pw))
    e_s = perpatch_logres2(ds_, gt, (ph, pw))
    m = np.isfinite(e_t) & np.isfinite(e_s)
    frac = float((e_t[m] < e_s[m]).mean())
    return {"auc_t": auc_t, "auc_s": auc_s, "absrel_t": ar_t, "absrel_s": ar_s,
            "frac_teacher_better": frac}


def main():
    os.makedirs(OUT, exist_ok=True)
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    base = M.get_base_vggt(student)
    M.reset_lora_(student)
    student.eval()
    rows = []
    for ds, scenes in SCENES.items():
        common.set_dataset(ds)
        dense = TAU[ds] > 0.55
        for scene in scenes:
            try:
                data = get_scene_data(scene)
                if ds == "eth3d":
                    from depth_anything_3.utils.constants import ETH3D_EVAL_DATA_ROOT
                    data.aux.gt_depth_files = [
                        os.path.join(ETH3D_EVAL_DATA_ROOT, scene, "ground_truth_depth",
                                     "dslr_images", os.path.basename(f))
                        for f in data.image_files]
            except Exception as e:
                print(f"[{ds}/{scene}] load failed: {e}", flush=True)
                continue
            import cv2
            orig_hw = cv2.imread(data.image_files[0]).shape[:2]
            N = len(data.image_files)
            for (tN, n_s) in COMBOS:
                if N < tN:
                    continue
                r = random.Random(stable_seed("combo", ds, scene, tN, n_s))
                for pi in range(N_PAIRS):
                    T, S = build_pair(N, tN, n_s, dense, r)
                    try:
                        met = combo_metrics(teacher, student, base, data, T, S, orig_hw)
                    except Exception as e:
                        print(f"[{ds}/{scene} {tN}:{n_s} p{pi}] failed: {e}", flush=True)
                        continue
                    if met is None:
                        continue
                    rows.append({"dataset": ds, "scene": scene, "combo": f"{tN}:{n_s}",
                                 "pair": pi, **met})
                    print(f"[{ds} {scene.split('/')[-1]} {tN}:{n_s} p{pi}] "
                          f"auc S={met['auc_s']:.3f} T={met['auc_t']:.3f} "
                          f"absrel S={met['absrel_s']:.4f} T={met['absrel_t']:.4f} "
                          f"fracT={met['frac_teacher_better']:.2f}", flush=True)
            del data
            torch.cuda.empty_cache()
    with open(os.path.join(OUT, "rows.json"), "w") as f:
        json.dump(rows, f)
    # pivot
    import collections
    import statistics as st
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["combo"])].append(r)
    lines = ["| dataset | combo | n | pose headroom (T-S) | depth headroom (S-T AbsRel) | fracT better |",
             "|---|---|---|---|---|---|"]
    for (ds, cb), rs in sorted(agg.items()):
        lines.append(f"| {ds} | {cb} | {len(rs)} | "
                     f"{st.mean(x['auc_t'] - x['auc_s'] for x in rs):+.3f} | "
                     f"{st.mean(x['absrel_s'] - x['absrel_t'] for x in rs):+.4f} | "
                     f"{st.mean(x['frac_teacher_better'] for x in rs):.2f} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(txt)
    print(txt, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
