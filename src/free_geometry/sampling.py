"""GT-free frame manifests. No model, torch, or benchmark imports."""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import SamplingConfig


def stable_seed(*parts):
    return int.from_bytes(
        hashlib.sha256("::".join(map(str, parts)).encode()).digest()[:8], "big"
    )


@dataclass
class SceneSource:
    scene: str
    image_files: list
    frame_ids: list
    dataset: str = "images"

    @classmethod
    def from_directory(cls, directory, scene=None):
        import re

        root = Path(directory).resolve()
        paths = sorted(
            (
                p
                for p in root.rglob("*")
                if p.suffix.lower()
                in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            ),
            key=lambda p: [
                int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(p))
            ],
        )
        return cls(
            scene or root.name,
            [str(p) for p in paths],
            [str(p.relative_to(root)) for p in paths],
        )


def compute_tau(paths):
    """Median adjacent SIFT match fraction; at most 40 adjacent pairs."""
    import cv2
    import numpy as np

    sift, cache = cv2.SIFT_create(nfeatures=400), {}

    def features(i):
        if i not in cache:
            im = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
            scale = min(1.0, 320 / max(im.shape))
            im = cv2.resize(
                im,
                (
                    max(1, round(im.shape[1] * scale)),
                    max(1, round(im.shape[0] * scale)),
                ),
            )
            cache[i] = sift.detectAndCompute(im, None)
        return cache[i]

    scores = []
    for i in np.linspace(0, len(paths) - 2, min(40, len(paths) - 1)).astype(int):
        ka, a = features(int(i))
        kb, b = features(int(i) + 1)
        if a is None or b is None or min(len(ka), len(kb)) < 10:
            scores.append(0.0)
            continue
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2)
        good = sum(
            len(m) == 2 and m[0].distance < 0.75 * m[1].distance for m in matches
        )
        scores.append(min(1.0, good / min(len(ka), len(kb))))
    return float(np.median(scores))


def assemble(shared, extras):
    n = len(shared) + len(extras)
    slots = [i * n // len(shared) for i in range(len(shared))]
    frames = [None] * n
    for slot, frame in zip(slots, shared):
        frames[slot] = frame
    rest = iter(sorted(extras))
    return [next(rest) if f is None else f for f in frames], slots


def sample_tasks(n, config, dense=False, scene="scene"):
    t, s = config.teacher_frames, config.student_frames
    if not t > s >= 2 or n < t:
        raise ValueError(
            f"insufficient frames or invalid ratio: N={n}, teacher={t}, student={s}"
        )
    count = config.train_tasks + config.probe_tasks
    if math.comb(n, s) < count:
        raise ValueError("not enough distinct shared combinations")
    rng = random.Random(stable_seed("frames", scene, config.seed))
    seen, tasks = set(), []
    for _ in range(max(10000, count * 100)):
        m = min(n, max(config.dense_candidates, 2 * t - s))
        pool = (
            [rng.randrange(i * n // m, (i + 1) * n // m) for i in range(m)]
            if dense
            else list(range(n))
        )
        shared = sorted(rng.sample(pool, s))
        key = tuple(shared)
        if key in seen:
            continue
        seen.add(key)
        extra_pool = [x for x in pool if x not in shared]
        a = rng.sample(extra_pool, t - s)
        remaining = [x for x in extra_pool if x not in a]
        b = rng.sample(remaining, min(t - s, len(remaining)))
        b += rng.sample(a, t - s - len(b))
        fa, sa = assemble(shared, a)
        fb, sb = assemble(shared, b)
        task = {
            "id": f"train{len(tasks)}"
            if len(tasks) < config.train_tasks
            else f"probe{len(tasks) - config.train_tasks}",
            "shared": shared,
            "teacher_a": fa,
            "slots_a": sa,
            "extras_a": sorted(a),
        }
        if len(tasks) < config.train_tasks:
            task.update(
                teacher_b=fb,
                slots_b=sb,
                extras_b=sorted(b),
                extras_overlap=len(set(a) & set(b)),
                ab_informative=fa != fb,
            )
        tasks.append(task)
        if len(tasks) == count:
            return tasks[: config.train_tasks], tasks[config.train_tasks :]
    raise RuntimeError(
        "sampling exhausted; refusing to pad with duplicate shared groups"
    )


def fingerprint(manifest):
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def build_manifest(source, config=None, tau_fn=compute_tau):
    from PIL import Image

    config = config or SamplingConfig()
    if len(source.frame_ids) != len(source.image_files) or len(
        set(source.frame_ids)
    ) != len(source.frame_ids):
        raise ValueError("frame IDs must be unique and correspond to image files")
    files, ids, bad = [], [], []
    for fid, path in zip(source.frame_ids, source.image_files):
        try:
            with Image.open(path) as im:
                im.load()
                if min(im.size) < 1:
                    raise ValueError("empty image")
            files.append(str(Path(path).resolve()))
            ids.append(fid)
        except (OSError, ValueError) as exc:
            bad.append({"frame_id": fid, "path": path, "reason": str(exc)})
    n = len(files)
    base = {
        "version": "unified-1",
        "scene": source.scene,
        "dataset": source.dataset,
        "image_files": files,
        "frame_ids": ids,
        "N_valid": n,
        "bad_files": bad,
        "sampling": asdict(config),
        "sift_called": False,
        "tau": None,
    }
    if n < config.teacher_frames:
        base.update(
            status="skipped",
            reason=f"N_valid={n} < teacher_frames={config.teacher_frames}",
        )
    else:
        called = n >= max(50, config.dense_min_frames) and config.sift_mode == "auto"
        tau = tau_fn(files) if called else None
        if called and (not math.isfinite(tau) or not 0 <= tau <= 1):
            raise ValueError("SIFT returned invalid overlap")
        dense = called and tau > config.sift_threshold
        train, probe = sample_tasks(n, config, dense, source.scene)
        ev = sorted(random.Random(42).sample(range(n), min(100, n)))
        base.update(
            status="ready",
            sift_called=called,
            tau=tau,
            strategy="dense" if dense else "random",
            train=train,
            probe=probe,
            eval_frames=ev,
        )
    base["fingerprint"] = fingerprint(base)
    validate_manifest(base)
    return base


def validate_manifest(m):
    if m.get("version") != "unified-1" or m.get("fingerprint") != fingerprint(m):
        raise ValueError(
            "manifest version/fingerprint mismatch; regenerate with fg prepare"
        )
    if m["status"] == "skipped":
        return m
    n = m["N_valid"]
    t = m["sampling"]["teacher_frames"]
    s = m["sampling"]["student_frames"]
    if n != len(m["image_files"]) or n != len(set(m["frame_ids"])):
        raise ValueError("manifest frame inventory mismatch")
    seen = set()
    for split, expected in (
        ("train", m["sampling"]["train_tasks"]),
        ("probe", m["sampling"]["probe_tasks"]),
    ):
        if len(m[split]) != expected:
            raise ValueError("task count mismatch")
        for task in m[split]:
            shared = task["shared"]
            key = tuple(sorted(shared))
            if len(set(shared)) != s or key in seen:
                raise ValueError("duplicate or malformed shared group")
            seen.add(key)
            for side in ("a", "b") if split == "train" else ("a",):
                frames, slots = task[f"teacher_{side}"], task[f"slots_{side}"]
                if (
                    len(frames) != t
                    or len(set(frames)) != t
                    or not all(type(i) is int and 0 <= i < n for i in frames)
                ):
                    raise ValueError("illegal teacher frames")
                if len(slots) != s or any(
                    type(i) is not int or not 0 <= i < t for i in slots
                ):
                    raise ValueError("illegal shared slots")
                if [frames[i] for i in slots] != shared or set(
                    task[f"extras_{side}"]
                ) != set(frames) - set(shared):
                    raise ValueError("frame/slot/extras mapping mismatch")
            if split == "train":
                overlap = len(set(task["extras_a"]) & set(task["extras_b"]))
                if (
                    overlap != max(0, 2 * t - s - n)
                    or task["extras_overlap"] != overlap
                ):
                    raise ValueError("A/B extras overlap is not minimal")
                if task["ab_informative"] != (task["teacher_a"] != task["teacher_b"]):
                    raise ValueError("invalid A/B informative flag")
    if len(set(m["eval_frames"])) != len(m["eval_frames"]) or not all(
        0 <= i < n for i in m["eval_frames"]
    ):
        raise ValueError("invalid evaluation frames")
    return m


def inspect_manifest(m):
    validate_manifest(m)
    if m["status"] == "skipped":
        return {"scene": m["scene"], "status": "skipped", "reason": m["reason"]}
    coverage = {fid: {"shared": 0, "extra": 0} for fid in m["frame_ids"]}
    for task in m["train"]:
        for i in task["shared"]:
            coverage[m["frame_ids"][i]]["shared"] += 1
        for side in ("a", "b"):
            for i in task[f"extras_{side}"]:
                coverage[m["frame_ids"][i]]["extra"] += 1
    return {
        k: m[k]
        for k in ("scene", "N_valid", "strategy", "tau", "sift_called", "fingerprint")
    } | {
        "unique_train_shared": len(m["train"]),
        "unique_probe_shared": len(m["probe"]),
        "tasks": m["train"] + m["probe"],
        "coverage": coverage,
        "bad_files": m["bad_files"],
    }
