#!/usr/bin/env python3
"""GT-free covisibility-filtered spread manifests (se24f): identical to se24c
but the overlap measure is classic SIFT feature-match fraction between frames
(matched keypoints / min(#kp_i, #kp_j)) - legal in TTA (no GT depth/poses).

Per scene: SIFT features for pool frames computed once; per pair, extras are
kept when their mean match-fraction with the 4 shared frames falls in
[0.1, 0.5] (same DUSt3R overlap-range criterion as se24c), backfilled otherwise.
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


def sift_feats(paths, max_side=320):
    import cv2
    sift = cv2.SIFT_create(nfeatures=500)
    feats = {}
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        sc = max_side / max(h, w)
        if sc < 1.0:
            img = cv2.resize(img, (int(w * sc), int(h * sc)))
        kp, des = sift.detectAndCompute(img, None)
        feats[i] = (kp, des)
    return feats


def match_frac(feats, i, j):
    import cv2
    kpi, di = feats[i]
    kpj, dj = feats[j]
    if di is None or dj is None or len(kpi) < 10 or len(kpj) < 10:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(di, dj, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    return len(good) / max(1, min(len(kpi), len(kpj)))


def pairs_se24f(pool, image_files, n, scene, ds, tag):
    r = random.Random(stable_seed("sub", "se24f", ds, scene, tag))
    N = len(pool)
    feats = sift_feats([image_files[i] for i in pool])
    res = []
    for k in range(n):
        m = min(24, N)
        spread_idx = sorted(set(int(i * N / m) for i in range(m)))
        spread = [pool[i] for i in spread_idx[:m]]
        s4 = sorted(r.sample(range(len(spread)), 4))
        student = [spread[i] for i in s4]
        rest = [f for f in spread if f not in set(student)]
        shared_pool_pos = [pool.index(f) for f in student]
        kept, dropped = [], []
        for f in rest:
            fp = pool.index(f)
            c = float(np.mean([match_frac(feats, fp, sp) for sp in shared_pool_pos]))
            (kept if 0.1 <= c <= 0.5 else dropped).append(f)
        rest = kept + dropped
        t = []
        for i, sv in enumerate(student):
            t.append(sv)
            if i < 3 and i < len(rest):
                t.append(rest[i])
        t += rest[3:]
        t = t[:m]
        assert [t[i] for i in [0, 2, 4, 6]] == student
        res.append({"teacher_frames": t, "student_frames": student})
    return res


def main():
    for ds, scenes in SUBSET.items():
        common.set_dataset(ds)
        src = load_manifest(f"artifacts/diagnostics/sub8_{ds}_se24/scene_manifest.json")
        out = {"run_root": f"artifacts/diagnostics/sub8_{ds}_se24f", "dataset": ds,
               "seed_scene_select": 43,
               "note": "8-scene subset, strategy se24f (GT-free SIFT-match filtered extras, frac in [0.1,0.5])",
               "scenes": {}}
        for scene in scenes:
            sc = src["scenes"][scene]
            ev = set(sc["eval32_frames"])
            sd = common.get_scene_data(scene)
            if ds == "eth3d":
                gtf = [os.path.join("workspace/benchmark_dataset/eth3d", scene,
                                    "ground_truth_depth", "dslr_images", os.path.basename(f))
                       for f in sd.image_files]
                ok = [i for i, p in enumerate(gtf) if os.path.exists(p)]
            else:
                from common import frames_with_gt_depth
                ok, _ = frames_with_gt_depth(scene)
            pool = [i for i in ok if i not in ev]
            out["scenes"][scene] = dict(sc)
            out["scenes"][scene]["train_pairs"] = pairs_se24f(pool, sd.image_files, 10, scene, ds, "tr")
            out["scenes"][scene]["probe_pairs"] = pairs_se24f(pool, sd.image_files, 2, scene, ds, "pr")
            del sd
        save_manifest(out, f"artifacts/diagnostics/sub8_{ds}_se24f/scene_manifest.json")
        print(ds, "se24f ok", flush=True)


if __name__ == "__main__":
    main()
