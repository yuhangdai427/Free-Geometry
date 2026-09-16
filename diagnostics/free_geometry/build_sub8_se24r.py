#!/usr/bin/env python3
"""se24r manifests: retrieval-embedding frame selection (GT-free, pose-free).

Per scene: embed every pool frame with the frozen VGGT's DINOv2 patch-embed
CLS token (x_norm_clstoken - a global frame descriptor), then farthest-point
sample 24 teacher frames in embedding space (coverage maximization), student =
4 random of them at slots [0,2,4,6].
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
from common import load_manifest, save_manifest, stable_seed  # noqa: E402
import modeling as M  # noqa: E402

SUBSET = {
    "scannetpp": ["7831862f02", "bde1e479ad"],
    "7scenes": ["chess", "office"],
    "hiroom": ["20241230/828738/cam_sampled_08", "20241230/828749/cam_sampled_12"],
    "eth3d": ["courtyard", "office"],
}


@torch.no_grad()
def embed_frames(student, image_files, pool, device="cuda"):
    """DINOv2 CLS embedding per pool frame (224x224 center input)."""
    import cv2
    agg = M.get_base_vggt(student).aggregator
    pe = agg.patch_embed
    embs = []
    B = 64
    for s in range(0, len(pool), B):
        imgs = []
        for i in pool[s:s + B]:
            img = cv2.imread(image_files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            imgs.append(img.astype(np.float32) / 255.0)
        t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(device)
        with torch.autocast(device_type="cuda", enabled=False):
            out = pe(t)
        e = out["x_norm_clstoken"].float().cpu()
        embs.append(e)
    E = torch.cat(embs, 0)
    return torch.nn.functional.normalize(E, dim=-1)


def fps_select(E, m, seed):
    """Farthest-point sampling of m points from E [N,C] (cosine distance)."""
    rng = np.random.default_rng(seed)
    N = E.shape[0]
    sel = [int(rng.integers(N))]
    dmin = 1.0 - (E @ E[sel[0]]).numpy()
    while len(sel) < m:
        nxt = int(np.argmax(dmin))
        sel.append(nxt)
        dmin = np.minimum(dmin, 1.0 - (E @ E[nxt]).numpy())
    return sorted(sel)


def main():
    student = M.load_student("cuda")
    for ds, scenes in SUBSET.items():
        common.set_dataset(ds)
        src = load_manifest(f"artifacts/diagnostics/sub8_{ds}_se24/scene_manifest.json")
        out = {"run_root": f"artifacts/diagnostics/sub8_{ds}_se24r", "dataset": ds,
               "seed_scene_select": 43,
               "note": "8-scene subset, strategy se24r (DINOv2-CLS FPS over embeddings; GT-free)",
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
            E = embed_frames(student, sd.image_files, pool)
            m = min(24, len(pool))
            t_idx = fps_select(E, m, stable_seed("fps", ds, scene) % (2**31))
            spread = [pool[i] for i in t_idx]
            r = random.Random(stable_seed("sub", "se24r", ds, scene))
            def mk(n, tag):
                rr = random.Random(stable_seed("sub", "se24r", ds, scene, tag))
                res = []
                for k in range(n):
                    s4 = sorted(rr.sample(range(len(spread)), 4))
                    student_f = [spread[i] for i in s4]
                    rest = [f for f in spread if f not in set(student_f)]
                    t = []
                    for i, sv in enumerate(student_f):
                        t.append(sv)
                        if i < 3:
                            t.append(rest[i])
                    t += rest[3:]
                    assert len(t) == m and [t[i] for i in [0, 2, 4, 6]] == student_f
                    res.append({"teacher_frames": t, "student_frames": student_f})
                return res
            out["scenes"][scene] = dict(sc)
            out["scenes"][scene]["train_pairs"] = mk(10, "tr")
            out["scenes"][scene]["probe_pairs"] = mk(2, "pr")
            print(f"[{ds}/{scene}] embedded {len(pool)} -> 24 FPS teacher frames", flush=True)
            del sd
        save_manifest(out, f"artifacts/diagnostics/sub8_{ds}_se24r/scene_manifest.json")
        print(ds, "se24r ok", flush=True)


if __name__ == "__main__":
    main()
