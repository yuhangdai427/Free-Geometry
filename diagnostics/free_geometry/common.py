"""Shared utilities for the Free-Geometry diagnostic bake-off (VGGT x ScanNet++).

Convention notes (audited 2026-09-09):
- ScanNet++ GT depth: uint16 PNG in millimetres, rendered in the RAW (distorted)
  camera frame at 1440x1920 -> load with resize-only, /1000, clamp at 5.0 m.
  Do NOT use ScanNetPP.load_image (undistort+ROI crop); it is inconsistent with
  the GT depth frame and unused anywhere else in the repo.
- Image preprocessing for the model mirrors VGGTFreeGeometryDataset: longest side
  -> 504, round to multiple of 14, /255 to [0,1], NO augmentation.
- patch_start_idx = 5 (1 camera token + 4 register tokens).
"""

import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "vggt"))

SCANNETPP_SCENE_LIST = [
    "09c1414f1b", "1ada7a0617", "21d970d8de", "286b55a2bf", "38d58a7a31",
    "3e8bba0176", "40aec5fffa", "578511c8a9", "5f99900f09", "7831862f02",
    "7bc286c1b6", "9071e139d9", "acd95847c5", "bcd2436daf", "bde1e479ad",
    "c4c04e6d6c", "c5439f4607", "cc5237fd77", "f3d64c30f8", "fb5a96b1a2",
]

MODEL_PATH = os.path.join(_REPO_ROOT, "model_weights", "VGGT-1B")
RUN_ROOT_DEFAULT = os.path.join(_REPO_ROOT, "artifacts", "diagnostics", "bakeoff_v1")

PATCH_START_IDX = 5
TAP_LAYERS = [4, 11, 17, 23]
STUDENT_INDICES = [0, 2, 4, 6]
MAX_DEPTH_M = 5.0
IMAGE_SIZE = 504

SEED_SCENE_SELECT = 43
SEED_EVAL = 43_000
SEED_TRAIN_PAIRS = 44_000
SEED_PROBE_PAIRS = 45_000


def stable_seed(*parts) -> int:
    key = "::".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


_CURRENT_DATASET = "scannetpp"


def set_dataset(name: str) -> None:
    global _CURRENT_DATASET
    _CURRENT_DATASET = name


def get_scene_data(scene: str):
    """Return dataset.get_data(scene) (addict Dict with image_files, aux...)."""
    if _CURRENT_DATASET == "7scenes":
        from depth_anything_3.bench.datasets.sevenscenes import SevenScenes
        return SevenScenes().get_data(scene)
    if _CURRENT_DATASET == "hiroom":
        from depth_anything_3.bench.datasets.hiroom import HiRoomDataset
        return HiRoomDataset().get_data(scene)
    if _CURRENT_DATASET == "eth3d":
        from depth_anything_3.bench.datasets.eth3d import ETH3D
        from depth_anything_3.utils.constants import ETH3D_EVAL_DATA_ROOT
        d = ETH3D().get_data(scene)
        d.aux.gt_depth_files = [
            os.path.join(ETH3D_EVAL_DATA_ROOT, scene, "ground_truth_depth",
                         "dslr_images", os.path.basename(f))
            for f in d.image_files]
        return d
    from depth_anything_3.bench.datasets.scannetpp import ScanNetPP

    ds = ScanNetPP()
    return ds.get_data(scene)


def gt_ixt_raw(scene_data, frames):
    """GT raw intrinsics for the given frames; falls back to model intrinsics
    (7Scenes/HiRoom have no separate raw-undistort intrinsics)."""
    aux = scene_data.aux
    src = aux["ixt_raw_list"] if "ixt_raw_list" in aux else scene_data.intrinsics
    return np.asarray(src)[frames]


def frames_with_gt_depth(scene: str) -> Tuple[List[int], object]:
    """Indices into scene_data.image_files whose GT depth PNG exists."""
    data = get_scene_data(scene)
    ok = []
    for i, p in enumerate(data.aux.gt_depth_files):
        if os.path.exists(p):
            ok.append(i)
    return ok, data


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_model(path: str, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Dataset-identical preprocessing, no augmentation. Returns [H,W,3] float32 in [0,1]."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    new_h = max(14, int(round(h * scale / 14) * 14))
    new_w = max(14, int(round(w * scale / 14) * 14))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return img.astype(np.float32) / 255.0


DATA_MAX_DEPTH = {"scannetpp": 5.0, "7scenes": 10.0, "hiroom": 100.0, "eth3d": 150.0}


def load_gt_depth(path: str, out_hw: Tuple[int, int]) -> np.ndarray:
    """GT depth -> metres, resize-only to out_hw=(H,W). NaN where invalid.
    scannetpp/7scenes: uint16 mm PNG; hiroom: uint16 PNG scaled to 100m;
    eth3d: raw float32 at full image resolution (shape read from the image)."""
    if _CURRENT_DATASET == "eth3d":
        img_path = path.replace(os.sep + "ground_truth_depth" + os.sep, os.sep + "images" + os.sep)
        im = cv2.imread(img_path)
        h, w = im.shape[:2]
        d = np.fromfile(path, dtype=np.float32).reshape(h, w)
        d = cv2.resize(d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
        d[(d <= 0) | ~np.isfinite(d) | (d > 150.0)] = np.nan
        return d
    if _CURRENT_DATASET == "hiroom":
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"Failed to load GT depth: {path}")
        d = d.astype(np.float32) / 65535.0 * 100.0
        d = cv2.resize(d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
        d[(d <= 0) | ~np.isfinite(d)] = np.nan
        return d
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise ValueError(f"Failed to load GT depth: {path}")
    d = d.astype(np.float32) / 1000.0
    d = cv2.resize(d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    max_d = DATA_MAX_DEPTH.get(_CURRENT_DATASET, MAX_DEPTH_M)
    invalid = (d <= 0) | ~np.isfinite(d) | (d > max_d)
    d[invalid] = np.nan
    return d


def e_depth(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, int]:
    """Scale-invariant log-depth error, ONE shared scale correction across views.

    pred, gt: [S,H,W] arrays (gt may contain NaN = invalid).
    Omega = GT-valid only, identical across interventions. Returns (E, n_valid).
    Non-finite or non-positive predictions inside Omega are a numeric failure:
    counted as a large penalty rather than shrinking Omega.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    omega = np.isfinite(gt) & (gt > 0)
    n = int(omega.sum())
    if n == 0:
        return float("nan"), 0
    bad = (~np.isfinite(pred) | (pred <= 0)) & omega
    if bad.any():
        return float("nan"), -int(bad.sum())  # negative n signals numeric failure
    r = np.log(pred[omega]) - np.log(gt[omega])
    c = r.mean()
    return float(np.mean((r - c) ** 2)), n


def build_scene_manifest(run_root: str, n_scenes: int = 6) -> Dict:
    """Select n dev scenes (seed 43) and fix eval/train/probe frame splits.
    Per-scene splits are stable-seeded per scene id, so enlarging n does not
    change the existing 6 scenes' splits."""
    rng = random.Random(SEED_SCENE_SELECT)
    scenes = sorted(rng.sample(SCANNETPP_SCENE_LIST, min(n_scenes, len(SCANNETPP_SCENE_LIST))))

    manifest = {"run_root": run_root, "seed_scene_select": SEED_SCENE_SELECT,
                "scenes": {}}
    for scene in scenes:
        ok, data = frames_with_gt_depth(scene)
        assert len(ok) >= 48, f"{scene}: only {len(ok)} frames with GT depth"
        ev_rng = random.Random(stable_seed(SEED_EVAL, scene))
        eval_frames = sorted(ev_rng.sample(ok, 32))
        pool = [i for i in ok if i not in set(eval_frames)]
        assert len(pool) >= 48, f"{scene}: adaptation pool too small ({len(pool)})"

        def sample_pairs(seed_tag: int, n: int, forbidden8: set, forbidden4: set):
            r = random.Random(stable_seed(seed_tag, scene))
            pairs = []
            tries = 0
            while len(pairs) < n and tries < 10000:
                tries += 1
                s8 = tuple(sorted(r.sample(pool, 8)))
                s4 = tuple(s8[i] for i in STUDENT_INDICES)
                if s8 in forbidden8 or s4 in forbidden4:
                    continue
                forbidden8.add(s8)
                forbidden4.add(s4)
                pairs.append({"teacher_frames": list(s8), "student_frames": list(s4)})
            assert len(pairs) == n, f"{scene}: could not sample {n} disjoint pairs"
            return pairs

        train_pairs = sample_pairs(SEED_TRAIN_PAIRS, 10, set(), set())
        probe_pairs = sample_pairs(
            SEED_PROBE_PAIRS, 2,
            {tuple(p["teacher_frames"]) for p in train_pairs},
            {tuple(p["student_frames"]) for p in train_pairs},
        )

        used = sorted(set(eval_frames)
                      | {f for p in train_pairs + probe_pairs for f in p["teacher_frames"]})
        manifest["scenes"][scene] = {
            "num_frames_total": len(data.image_files),
            "num_frames_with_gt_depth": len(ok),
            "eval32_frames": eval_frames,
            "adaptation_pool_size": len(pool),
            "train_pairs": train_pairs,
            "probe_pairs": probe_pairs,
            "file_sha256": {
                str(i): sha256_file(data.image_files[i]) for i in used
            },
        }
    return manifest


def save_manifest(manifest: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_disjoint(manifest: Dict) -> None:
    """D0 gate: eval frames must not appear in any train/probe pair."""
    for scene, sc in manifest["scenes"].items():
        ev = set(sc["eval32_frames"])
        for p in sc["train_pairs"] + sc["probe_pairs"]:
            assert not (ev & set(p["teacher_frames"])), f"{scene}: eval frame leak"
        s8 = set()
        for p in sc["train_pairs"]:
            t = tuple(p["teacher_frames"])
            assert len(set(t)) == 8 and t not in s8
            s8.add(t)
            assert tuple(p["student_frames"]) == tuple(p["teacher_frames"][i] for i in STUDENT_INDICES)
