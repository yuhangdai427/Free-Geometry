#!/usr/bin/env python3
"""Step-1 builder: se8/se16/se24c manifests for the 8-scene subset.

se-N: teacher = N frames evenly spread over the pool (int(i*M/N)); student = 4
random of them at slots [0,2,4,6].
se24c: same as se24 but the 20 extra frames are filtered by covisibility with
the 4 shared frames in [0.1, 0.5] (DUSt3R overlap-range criterion), using the
repo's precomputed raw_covisibility matrices (artifacts/covisibility_*).
Falls back to unfiltered spread when a scene has no matrix or too few pass.
"""

import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common  # noqa: E402
from common import load_manifest, save_manifest, stable_seed  # noqa: E402

SUBSET = {
    "scannetpp": ["7831862f02", "bde1e479ad"],
    "7scenes": ["chess", "office"],
    "hiroom": ["20241230/828738/cam_sampled_08", "20241230/828749/cam_sampled_12"],
    "eth3d": ["courtyard", "office"],
}
COV_ROOT = {
    "scannetpp": "artifacts/covisibility_scannetpp_all_frames/scannetpp",
    "7scenes": "artifacts/covisibility_all_datasets_0025/7scenes",
    "hiroom": "artifacts/covisibility_all_datasets_0025/hiroom",
    "eth3d": "artifacts/covisibility_eth3d_all/eth3d",
}


def frame_pos_mapper(ds, scene, z):
    """Map covisibility-matrix row i -> frame index in scene_data.image_files."""
    fids = [str(f) for f in z["frame_ids"]]
    if ds == "7scenes":
        return {i: int(f.split("-")[1]) for i, f in enumerate(fids)}
    if ds == "scannetpp":
        return {i: int(f.split("_")[1]) for i, f in enumerate(fids)}
    # hiroom/eth3d: frame name -> position in sorted image list
    sd = common.get_scene_data(scene)
    name2pos = {os.path.basename(p).split(".")[0]: i for i, p in enumerate(sd.image_files)}
    return {i: name2pos[f] for i, f in enumerate(fids) if f in name2pos}


def spread_pairs(pool, n, teacher_n, scene, ds, tag, cov_filter=False, z=None, fpos=None):
    r = random.Random(stable_seed("sub", f"se{teacher_n}{'c' if cov_filter else ''}", ds, scene, tag))
    res = []
    N = len(pool)
    for k in range(n):
        m = min(teacher_n, N)
        spread_idx = sorted(set(int(i * N / m) for i in range(m)))
        spread = [pool[i] for i in spread_idx[:m]]
        s4 = sorted(r.sample(range(len(spread)), 4))
        student = [spread[i] for i in s4]
        rest = [f for f in spread if f not in set(student)]
        if cov_filter and z is not None and fpos:
            inv = {v: k2 for k2, v in fpos.items()}  # frame idx -> row
            shared_rows = [inv[f] for f in student if f in inv]
            kept, dropped = [], []
            for f in rest:
                if f not in inv or not shared_rows:
                    dropped.append(f)
                    continue
                row = z["raw_covisibility"][inv[f]]
                c = float(np.mean([row[sr] for sr in shared_rows]))
                (kept if 0.1 <= c <= 0.5 else dropped).append(f)
            # keep teacher size: backfill from dropped (far) then kept pool order
            rest = kept + [f for f in dropped] + [f for f in spread if f not in set(kept) and f not in set(dropped)]
        t = []
        for i, sv in enumerate(student):
            t.append(sv)
            if i < 3 and i < len(rest):
                t.append(rest[i])
        t += rest[3:]
        t = t[:m]
        assert [t[i] for i in [0, 2, 4, 6]] == student, (ds, scene, k)
        res.append({"teacher_frames": t, "student_frames": student})
    return res


def main():
    for ds, scenes in SUBSET.items():
        common.set_dataset(ds)
        for strat, teacher_n, covf in [("se8", 8, False), ("se16", 16, False), ("se24c", 24, True)]:
            src = load_manifest(f"artifacts/diagnostics/sub8_{ds}_se24/scene_manifest.json")
            out = {"run_root": f"artifacts/diagnostics/sub8_{ds}_{strat}", "dataset": ds,
                   "seed_scene_select": 43,
                   "note": f"8-scene subset, strategy {strat}"
                           + (" (covisibility-filtered extras, raw_cov in [0.1,0.5])" if covf else ""),
                   "scenes": {}}
            for scene in scenes:
                sc = src["scenes"][scene]
                ev = set(sc["eval32_frames"])
                pool = None
                # recover pool from the se24 manifest: all GT frames minus eval
                from common import frames_with_gt_depth
                if ds == "eth3d":
                    sd = common.get_scene_data(scene)
                    gtf = [os.path.join("workspace/benchmark_dataset/eth3d", scene,
                                        "ground_truth_depth", "dslr_images", os.path.basename(f))
                           for f in sd.image_files]
                    ok = [i for i, p in enumerate(gtf) if os.path.exists(p)]
                else:
                    ok, _ = frames_with_gt_depth(scene)
                pool = [i for i in ok if i not in ev]
                z = None
                fpos = None
                if covf:
                    p = os.path.join(COV_ROOT[ds], scene.replace("/", "__") + ".npz")
                    if os.path.exists(p):
                        z = np.load(p, allow_pickle=True)
                        fpos = frame_pos_mapper(ds, scene, z)
                out["scenes"][scene] = dict(sc)
                out["scenes"][scene]["train_pairs"] = spread_pairs(pool, 10, teacher_n, scene, ds, "tr", covf, z, fpos)
                out["scenes"][scene]["probe_pairs"] = spread_pairs(pool, 2, teacher_n, scene, ds, "pr", covf, z, fpos)
            save_manifest(out, f"artifacts/diagnostics/sub8_{ds}_{strat}/scene_manifest.json")
        print(ds, "se8/se16/se24c ok")


if __name__ == "__main__":
    main()
