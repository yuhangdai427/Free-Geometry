#!/usr/bin/env python3
"""Final-protocol (docs/TTA_PROTOCOL_v1_2026-09-13.md) GT-free manifests.

Dataset-agnostic rules (only N and tau are measured on the spot):
- pool = ALL N frames of the scene (transductive; zero GT anywhere in selection)
- tau  = median adjacent-frame SIFT match fraction (<=40 equidistant adjacent
  pairs, images shrunk to 320px, SIFT 400 feats, ratio test 0.75)
- teacher_N = 16 if N>=16 else 8 (N<8: scene dropped, recorded in report)
- tau > 0.55 (dense video): per-pair random-offset equidistant superset
  M=min(N, teacher_N*3//2); shared = sup[::M//4][:4]; extras chosen from the
  rest, soft-filtered: prefer mean SIFT match fraction vs the 4 shared frames
  in [0.1, 0.5] (DUSt3R overlap range, GT-free), backfill with out-of-range.
- tau <= 0.55: pure random teacher_N frames (sorted); shared = sorted[::teacher_N//4]
- teacher list layout: shared frames at slots [0,2,4,6] (STUDENT_INDICES, hard
  requirement of train_arms/common.verify), extras ascending in remaining slots.
- eval32_frames: N>=100 -> benchmark-100 (random.seed(42); shuffle(range(N));
  sorted(idx[:100]), aligned with src/vggt/vggt/bench/evaluator.py); N<100 -> all.
- 10 train pairs + 2 probe pairs, stable-seeded per (dataset, scene).

Run from repo root:  python diagnostics/free_geometry/build_final_manifest.py
"""

import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import common  # noqa: E402
from common import SCANNETPP_SCENE_LIST, get_scene_data, save_manifest, stable_seed  # noqa: E402

OUT_ROOT = os.path.join(_REPO, "artifacts", "diagnostics", "final_protocol")

SEVEN_SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
HIROOM_LIST = os.path.join(_REPO, "workspace", "benchmark_dataset", "hiroom",
                           "selected_scene_list_val.txt")

DATASETS = {
    "scannetpp": list(SCANNETPP_SCENE_LIST),
    "7scenes": list(SEVEN_SCENES),
    "hiroom": [l.strip() for l in open(HIROOM_LIST).read().splitlines() if l.strip()],
    "eth3d": None,  # filled from ETH3D_SCENES
}

TAU_THRESHOLD = 0.55
OVERLAP_LO, OVERLAP_HI = 0.1, 0.5
N_TRAIN, N_PROBE = 10, 2


# ---------------------------------------------------------------- SIFT utils
def sift_one(path, max_side=320):
    import cv2
    sift = cv2.SIFT_create(nfeatures=400)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    sc = max_side / max(h, w)
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)))
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def match_frac(feats_i, feats_j):
    import cv2
    kpi, di = feats_i
    kpj, dj = feats_j
    if di is None or dj is None or len(kpi) < 10 or len(kpj) < 10:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(di, dj, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    return len(good) / max(1, min(len(kpi), len(kpj)))


class SiftCache:
    def __init__(self, image_files):
        self.image_files = image_files
        self.cache = {}

    def feats(self, i):
        if i not in self.cache:
            self.cache[i] = sift_one(self.image_files[i])
        return self.cache[i]

    def frac(self, i, j):
        return match_frac(self.feats(i), self.feats(j))


def compute_tau(image_files):
    """Median adjacent-frame SIFT match fraction over <=40 equidistant pairs."""
    N = len(image_files)
    if N < 2:
        return 0.0
    sc = SiftCache(image_files)
    n_pairs = min(40, N - 1)
    starts = np.linspace(0, N - 2, n_pairs).astype(int)
    fr = [sc.frac(int(i), int(i) + 1) for i in starts]
    return float(np.median(fr))


# ---------------------------------------------------------------- pair build
def assemble_teacher_list(shared, extras):
    """Shared (ascending) at slots [0,2,4,6]; extras (ascending) fill the rest."""
    teacher = [None] * (len(shared) + len(extras))
    for slot, s in zip([0, 2, 4, 6], shared):
        teacher[slot] = s
    rest = [i for i in range(len(teacher)) if i not in (0, 2, 4, 6)]
    for slot, e in zip(rest, extras):
        teacher[slot] = e
    return teacher


def build_pairs(N, teacher_N, dense, sc, rng):
    """One pair: (teacher_frames, student_frames)."""
    if dense:
        M = min(N, teacher_N * 3 // 2)
        step = N / M
        off = rng.uniform(0, step)
        sup = sorted(set(int(off + i * step) for i in range(M)))
        sup = sup[:M]
        stride = max(1, M // 4)
        shared = sup[::stride][:4]
        rest = [f for f in sup if f not in set(shared)]
        in_range, out_range = [], []
        for f in rest:
            c = float(np.mean([sc.frac(f, s) for s in shared]))
            (in_range if OVERLAP_LO <= c <= OVERLAP_HI else out_range).append(f)
        extras = (in_range + out_range)[:teacher_N - 4]
        extras = sorted(extras)
    else:
        t = sorted(rng.sample(range(N), teacher_N))
        stride = teacher_N // 4
        shared = t[::stride][:4]
        extras = [f for f in t if f not in set(shared)]
    shared = sorted(shared)
    teacher = assemble_teacher_list(shared, extras)
    return teacher, shared


def build_scene_entry(ds, scene):
    common.set_dataset(ds)
    data = get_scene_data(scene)
    image_files = list(data.image_files)
    N = len(image_files)
    entry = {"num_frames_total": N, "file_sha256": {}}
    if N < 8:
        return None, {"scene": scene, "N": N, "dropped": "N<8"}
    teacher_N = 16 if N >= 16 else 8
    tau = compute_tau(image_files)
    dense = tau > TAU_THRESHOLD
    sc = SiftCache(image_files) if dense else None

    def sample(tag, n, forb_t, forb_s):
        r = random.Random(stable_seed("final", tag, ds, scene))
        pairs, tries = [], 0
        used_frames = set()
        while len(pairs) < n and tries < 10000:
            tries += 1
            teacher, shared = build_pairs(N, teacher_N, dense, sc, r)
            key_t, key_s = tuple(sorted(teacher)), tuple(shared)
            if key_t in forb_t or key_s in forb_s:
                continue
            forb_t.add(key_t)
            forb_s.add(key_s)
            used_frames |= set(teacher)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
        if len(pairs) < n:
            # Tiny scenes (e.g. N == teacher_N): distinct-pair space exhausted.
            # Fallback = coverage-greedy: among 64 fresh candidates keep the one
            # covering the most previously-unused frames (duplicates only when
            # unavoidable; per-step random masks still differ by pair index).
            while len(pairs) < n:
                best, best_gain = None, -1
                for _ in range(64):
                    teacher, shared = build_pairs(N, teacher_N, dense, sc, r)
                    gain = len(set(teacher) - used_frames)
                    if gain > best_gain:
                        best, best_gain = (teacher, shared), gain
                    if gain == teacher_N:
                        break
                teacher, shared = best
                used_frames |= set(teacher)
                pairs.append({"teacher_frames": teacher, "student_frames": list(shared)})
            print(f"  [relax] {scene}: pair space exhausted, coverage-greedy reuse "
                  f"(pool covered {len(used_frames)}/{N})", flush=True)
        return pairs

    train_pairs = sample("train", N_TRAIN, set(), set())
    probe_pairs = sample("probe", N_PROBE,
                         {tuple(sorted(p["teacher_frames"])) for p in train_pairs},
                         {tuple(p["student_frames"]) for p in train_pairs})

    if N >= 100:
        r = random.Random(42)
        idx = list(range(N))
        r.shuffle(idx)
        eval_frames = sorted(idx[:100])
    else:
        eval_frames = list(range(N))

    n_gt = sum(1 for p in data.aux.gt_depth_files if os.path.exists(p))
    entry.update({
        "num_frames_with_gt_depth": n_gt,
        "adaptation_pool_size": N,
        "teacher_N": teacher_N,
        "tau": tau,
        "strategy": "dense_equidistant_sift" if dense else "random",
        "eval32_frames": eval_frames,
        "train_pairs": train_pairs,
        "probe_pairs": probe_pairs,
    })
    info = {"scene": scene, "N": N, "tau": tau, "teacher_N": teacher_N,
            "strategy": entry["strategy"], "eval_n": len(eval_frames),
            "gt_coverage": n_gt / N}
    return entry, info


def main():
    from depth_anything_3.utils.constants import ETH3D_SCENES
    DATASETS["eth3d"] = list(ETH3D_SCENES)
    reports = {}
    for ds, scenes in DATASETS.items():
        run_root = os.path.join(OUT_ROOT, ds)
        manifest = {"dataset": ds, "run_root": run_root, "seed_scene_select": 43,
                    "note": "final protocol v1 (2026-09-13): GT-free, pool=all frames, "
                            "tau-dispatched selection, teacher16/8, shared@[0,2,4,6], "
                            "benchmark-100/allv eval frames",
                    "scenes": {}}
        infos = []
        for scene in scenes:
            entry, info = build_scene_entry(ds, scene)
            infos.append(info)
            if entry is not None:
                manifest["scenes"][scene] = entry
            print(f"[{ds}] {info}", flush=True)
        save_manifest(manifest, os.path.join(run_root, "scene_manifest.json"))
        reports[ds] = infos
    with open(os.path.join(OUT_ROOT, "manifest_report.json"), "w") as f:
        json.dump(reports, f, indent=2)
    lines = ["# final_protocol manifest report", ""]
    for ds, infos in reports.items():
        lines.append(f"## {ds}")
        lines.append("| scene | N | tau | strategy | teacher_N | eval_n | gt_cov |")
        lines.append("|---|---|---|---|---|---|---|")
        for i in infos:
            if "dropped" in i:
                lines.append(f"| {i['scene']} | {i['N']} | - | DROPPED {i['dropped']} | - | - | - |")
            else:
                lines.append(f"| {i['scene']} | {i['N']} | {i['tau']:.3f} | {i['strategy']} "
                             f"| {i['teacher_N']} | {i['eval_n']} | {i['gt_coverage']:.2f} |")
        lines.append("")
    with open(os.path.join(OUT_ROOT, "manifest_report.md"), "w") as f:
        f.write("\n".join(lines))
    print("DONE", os.path.join(OUT_ROOT, "manifest_report.md"))


if __name__ == "__main__":
    main()
