#!/usr/bin/env python3
"""Headroom probe: is there anything to distill? Per dataset x frame-selection
strategy, measure the teacher-vs-student gap WITHOUT any training.

Per pair: teacher forward (8v or 24v) and student 4v forward (frozen baseline).
Metrics on the 4 shared views (pair-wide scale):
- AbsRel teacher vs GT / student vs GT -> gap
- pose AUC@3 teacher / student vs GT -> gap
- per-patch fraction where teacher depth beats student (the distillable share)

Strategies:
- rand8:   random 8 frames, student = slots [0,2,4,6]  (current main recipe)
- seq8s4:  8-frame window at stride 4 in file/video order (sparse-consecutive)
- selfevo: teacher = 24 frames spread evenly over the scene, student = 4 random
           frames of those 24 (SelfEvo-style frame dropping, long context)

No TSDF fusion (dataset-universal, fast). Output: artifacts/diagnostics/headroom_probe/summary.md
"""

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common  # noqa: E402
from common import STUDENT_INDICES, load_manifest, stable_seed  # noqa: E402
import modeling as M  # noqa: E402
from depth_metrics import frame_metrics  # noqa: E402

OUT = "artifacts/diagnostics/headroom_probe"
DATASETS = {
    "scannetpp": ["1ada7a0617", "38d58a7a31", "7831862f02", "7bc286c1b6", "9071e139d9", "bde1e479ad"],
    "7scenes": ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
    "hiroom": ["20241230/828738/cam_sampled_08", "20241230/828744/cam_sampled_04",
               "20241230/828745/cam_sampled_05", "20241230/828749/cam_sampled_12",
               "20241230/828750/cam_sampled_08", "20241230/828757/cam_sampled_06"],
    "eth3d": ["courtyard", "electro", "kicker", "pipes", "office", "relief"],
}
K_PAIRS = 6


def load_gt_any(scene_data, frame_idx, out_hw, orig_hw=None):
    ds = common._CURRENT_DATASET
    path = scene_data.aux.gt_depth_files[frame_idx]
    if ds == "eth3d":
        d = np.fromfile(path, dtype=np.float32)
        h, w = orig_hw
        d = d.reshape(h, w)
        import cv2
        d = cv2.resize(d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
        d[(d <= 0) | ~np.isfinite(d) | (d > 150.0)] = np.nan
        return d
    if ds == "hiroom":
        import cv2
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        d = d.astype(np.float32) / 65535.0 * 100.0
        d = cv2.resize(d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
        d[(d <= 0) | ~np.isfinite(d)] = np.nan
        return d
    return common.load_gt_depth(path, out_hw)


def sample_pairs(fnames, n, strategy, scene, tag, gt_ok):
    """Return list of (teacher_frames, student_frames)."""
    import random
    rng = random.Random(stable_seed("hr", scene, tag))
    N = len(fnames)
    pairs = []
    for k in range(n):
        if strategy == "rand8":
            if N < 8:
                continue
            t8 = sorted(rng.sample(range(N), 8))
            pairs.append((t8, [t8[i] for i in STUDENT_INDICES]))
        elif strategy == "seq8s4":
            s = 4
            if N < 8 * s:
                s = max(1, (N - 1) // 7)
            hi = N - 7 * s
            if hi < 1:
                continue
            st0 = rng.randrange(hi)
            t8 = [st0 + i * s for i in range(8)]
            pairs.append((t8, [t8[i] for i in STUDENT_INDICES]))
        elif strategy == "selfevo24":
            m = min(24, N)
            spread = sorted(rng.sample(range(N), m)) if m < N else list(range(N))
            # evenly spread alternative: every N//m-th
            spread = sorted(set(int(i * N / m) for i in range(m)))[:m]
            if len(spread) < 5:
                continue
            s4 = sorted(rng.sample(range(len(spread)), 4))
            pairs.append((spread, [spread[i] for i in s4]))
    return pairs


@torch.no_grad()
def pair_metrics(teacher, student, base, scene_data, t_frames, s_frames, orig_hw):
    images8, images4 = M.load_pair_images(scene_data, t_frames)
    tfeats, tpsi = M.aggregator_all(teacher, images8)
    sfeats, spsi = student._get_aggregator()(images4)
    slots_t = [None] * 24
    for l in [4, 11, 17, 23]:
        slots_t[l] = tfeats[l][:, [t_frames.index(f) for f in s_frames]].float().contiguous()
    dt, _ = M.replay_depth_nograd(base, slots_t, images4, tpsi)
    dt = dt.squeeze(0).squeeze(-1).float().cpu().numpy()
    sfeats_f = [f.float() for f in sfeats]
    ds_, _ = M.replay_depth_nograd(base, sfeats_f, images4, spsi)
    ds_ = ds_.squeeze(0).squeeze(-1).float().cpu().numpy()
    pose_t = M.replay_camera_nograd(base, slots_t)
    pose_s = M.replay_camera_nograd(base, sfeats_f)
    gt_ext = np.asarray(scene_data.extrinsics)[s_frames]
    auc_t = M.pose_auc(pose_t.float(), gt_ext)["auc03"]
    auc_s = M.pose_auc(pose_s.float(), gt_ext)["auc03"]
    H, W = dt.shape[-2:]
    gt4 = np.stack([load_gt_any(scene_data, f, (H, W), orig_hw) for f in s_frames], 0)
    valid = np.isfinite(gt4) & (gt4 > 0)
    if valid.sum() < 1000:
        return None
    def scaled_absrel(d):
        r = np.log(np.clip(d, 1e-6, None)) - np.log(np.clip(gt4, 1e-6, None))
        c = r[valid].mean()
        ds2 = d * np.exp(-c)
        vals = []
        for k in range(4):
            fm = frame_metrics(ds2[k], gt4[k])
            if fm:
                vals.append(fm["absrel"])
        return float(np.mean(vals))
    ar_t, ar_s = scaled_absrel(dt), scaled_absrel(ds_)
    # per-patch teacher-better fraction (patch_interp convention)
    from train_arms import perpatch_logres2
    ph, pw = H // 14, W // 14
    e_t = perpatch_logres2(dt, gt4, (ph, pw))
    e_s = perpatch_logres2(ds_, gt4, (ph, pw))
    m = np.isfinite(e_t) & np.isfinite(e_s)
    frac_better = float((e_t[m] < e_s[m]).mean())
    return {"absrel_t": ar_t, "absrel_s": ar_s, "auc_t": auc_t, "auc_s": auc_s,
            "frac_teacher_better": frac_better}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_glob", nargs="*", default=None,
                    help="limit to these manifest dirs (default: all sub8_*)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    teacher = M.load_teacher("cuda")
    student = M.load_student("cuda")
    base = M.get_base_vggt(student)
    M.reset_lora_(student)
    student.eval()
    rows = []
    import glob
    mdirs = []
    for g in (args.dir_glob or ["artifacts/diagnostics/sub8_*"]):
        mdirs += glob.glob(g)
    for mdir in sorted(set(mdirs)):
        manifest = load_manifest(os.path.join(mdir, "scene_manifest.json"))
        ds = manifest["dataset"]
        strat = manifest["note"].split("strategy ")[-1]
        common.set_dataset(ds)
        for scene, sc in sorted(manifest["scenes"].items()):
            try:
                scene_data = common.get_scene_data(scene)
                if ds == "eth3d":
                    scene_data.aux.gt_depth_files = [
                        os.path.join("workspace/benchmark_dataset/eth3d", scene,
                                     "ground_truth_depth", "dslr_images", os.path.basename(f))
                        for f in scene_data.image_files]
            except Exception as e:
                print(f"[{ds}/{scene}] get_data failed: {e}", flush=True)
                continue
            import cv2
            orig_hw = cv2.imread(scene_data.image_files[0]).shape[:2]
            for pi, pair in enumerate(sc["train_pairs"][:4] + sc["probe_pairs"]):
                tf, sf = pair["teacher_frames"], pair["student_frames"]
                try:
                    r = pair_metrics(teacher, student, base, scene_data, tf, sf, orig_hw)
                except Exception as e:
                    print(f"[{ds}/{scene}/{strat}/p{pi}] failed: {e}", flush=True)
                    continue
                if r is None:
                    continue
                rows.append({"dataset": ds, "scene": scene, "strategy": strat, "pair": pi, **r})
                print(f"[{ds} {scene.split('/')[-1]} {strat} p{pi}] "
                      f"absrel S={r['absrel_s']:.4f} T={r['absrel_t']:.4f} "
                      f"auc S={r['auc_s']:.3f} T={r['auc_t']:.3f} "
                      f"fracTbetter={r['frac_teacher_better']:.2f}", flush=True)
            del scene_data
            torch.cuda.empty_cache()
    import collections, statistics as st
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["strategy"])].append(r)
    lines = ["# headroom probe: teacher-student gap without training", "",
             "| dataset | strategy | n | AbsRel S | AbsRel T | gap | AUC@3 S | AUC@3 T | gap | frac patches T better |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (ds, strat), rs in sorted(agg.items()):
        lines.append(f"| {ds} | {strat} | {len(rs)} | {st.mean(x['absrel_s'] for x in rs):.4f} | "
                     f"{st.mean(x['absrel_t'] for x in rs):.4f} | {st.mean(x['absrel_s'] - x['absrel_t'] for x in rs):+.4f} | "
                     f"{st.mean(x['auc_s'] for x in rs):.3f} | {st.mean(x['auc_t'] for x in rs):.3f} | "
                     f"{st.mean(x['auc_t'] - x['auc_s'] for x in rs):+.3f} | "
                     f"{st.mean(x['frac_teacher_better'] for x in rs):.3f} |")
    txt = "\n".join(lines) + "\n"
    tag = ("_".join(g.rsplit("_", 1)[-1] for g in (args.dir_glob or ["full"]))) if args.dir_glob else "full"
    sfx = "" if tag == "full" else f"_{tag}"
    with open(os.path.join(OUT, f"summary{sfx}.md"), "w") as f:
        f.write(txt)
    with open(os.path.join(OUT, f"rows{sfx}.json"), "w") as f:
        json.dump(rows, f)
    print(txt, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
