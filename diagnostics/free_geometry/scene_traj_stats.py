"""GT camera-trajectory statistics for problem scenes (7scenes + scannetpp).

Read-only analysis. Pose conventions follow the benchmark loaders:
- 7scenes: frame-XXXXXX.pose.txt is a 4x4 camera-to-world matrix (np.loadtxt),
  see src/depth_anything_3/bench/datasets/sevenscenes.py.
- scannetpp: COLMAP model in merge_dslr_iphone/colmap/sparse_render_rgb read via
  depth_anything_3.utils.read_write_model.read_model; extrinsic is world-to-camera
  ([qvec2rotmat | tvec]), c2w = inv(w2c). Only iphone frames are used, matching the
  ScanNetPP benchmark loader (src/depth_anything_3/bench/datasets/scannetpp.py);
  dslr/render_rgb subsets are not part of the evaluated trajectory.

Per scene (all available frames, sorted by filename index):
  N, camera-center bbox spans, median adjacent translation step (m),
  median adjacent rotation angle (deg), median rot/step ratio (deg/m),
  revisit count = #frame pairs with center distance < 0.3 m and index gap > 50.

Usage: python diagnostics/free_geometry/scene_traj_stats.py
Prints two markdown tables and writes artifacts/diagnostics/final_protocol/SCENE_TRAJ_STATS.md.
"""

import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

SEVENSCENES_ROOT = os.path.join(ROOT, "workspace/benchmark_dataset/7scenes/7Scenes")
SCANNETPP_ROOT = os.path.join(ROOT, "workspace/benchmark_dataset/scannetpp")
OUT_MD = os.path.join(ROOT, "artifacts/diagnostics/final_protocol/SCENE_TRAJ_STATS.md")

SEVENSCENES_SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
SCANNETPP_WOUND = ["21d970d8de", "acd95847c5", "c4c04e6d6c", "bcd2436daf",
                   "c5439f4607", "1ada7a0617", "7bc286c1b6"]
SCANNETPP_HEALTHY = ["f3d64c30f8", "fb5a96b1a2", "7831862f02", "cc5237fd77"]

REVISIT_DIST_M = 0.3
REVISIT_MIN_GAP = 50


def load_7scenes_c2w(scene):
    """All seq-*/frame-*.pose.txt (c2w 4x4), sorted by (seq, frame index)."""
    scene_dir = os.path.join(SEVENSCENES_ROOT, scene)
    entries = []
    for seq in sorted(os.listdir(scene_dir)):
        seq_dir = os.path.join(scene_dir, seq)
        if not (os.path.isdir(seq_dir) and seq.startswith("seq-")):
            continue
        for fname in os.listdir(seq_dir):
            if fname.endswith(".pose.txt"):
                frame_idx = int(fname.split("-")[1].split(".")[0])
                entries.append((seq, frame_idx, os.path.join(seq_dir, fname)))
    entries.sort(key=lambda e: (e[0], e[1]))
    poses, n_bad = [], 0
    for _, _, path in entries:
        try:
            m = np.loadtxt(path)
            if m.shape == (4, 4) and np.isfinite(m).all():
                poses.append(m)
            else:
                n_bad += 1
        except Exception:
            n_bad += 1
    if not poses:
        raise RuntimeError(f"no readable 4x4 poses under {scene_dir}")
    return np.stack(poses), n_bad


def load_scannetpp_c2w(scene):
    """iPhone frames from the COLMAP model; c2w = inv(w2c). Sorted by frame index."""
    from depth_anything_3.utils.read_write_model import read_model

    colmap_dir = os.path.join(SCANNETPP_ROOT, scene, "merge_dslr_iphone/colmap/sparse_render_rgb")
    _, images, _ = read_model(colmap_dir)
    entries = []
    for img in images.values():
        if "iphone" not in img.name:
            continue
        frame_idx = int(os.path.basename(img.name).replace("frame_", "").split(".")[0])
        entries.append((frame_idx, img))
    entries.sort(key=lambda e: e[0])
    if not entries:
        raise RuntimeError(f"no iphone frames in {colmap_dir}")
    poses = []
    for _, img in entries:
        w2c = np.eye(4)
        w2c[:3, :3] = img.qvec2rotmat()
        w2c[:3, 3] = img.tvec
        poses.append(np.linalg.inv(w2c))
    return np.stack(poses), 0


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def scene_stats(c2w):
    centers = c2w[:, :3, 3]
    rots = c2w[:, :3, :3]
    n = len(c2w)
    bbox = centers.max(axis=0) - centers.min(axis=0)

    steps, angs, ratios = [], [], []
    for i in range(n - 1):
        step = float(np.linalg.norm(centers[i + 1] - centers[i]))
        ang = rot_angle_deg(rots[i].T @ rots[i + 1])
        steps.append(step)
        angs.append(ang)
        ratios.append(ang / step if step > 1e-6 else np.nan)

    dmat = np.linalg.norm(centers[None, :, :] - centers[:, None, :], axis=-1)
    gap = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    revisits = int(((dmat < REVISIT_DIST_M) & (gap > REVISIT_MIN_GAP)).sum() // 2)

    return {
        "n": n,
        "bbox": bbox,
        "med_step": float(np.median(steps)),
        "med_rot": float(np.median(angs)),
        "med_ratio": float(np.nanmedian(ratios)),
        "revisits": revisits,
    }


def fmt_row(scene, s, note=""):
    b = s["bbox"]
    row = (f"| {scene} | {s['n']} | {b[0]:.2f} / {b[1]:.2f} / {b[2]:.2f} | "
           f"{s['med_step']:.4f} | {s['med_rot']:.2f} | {s['med_ratio']:.0f} | "
           f"{s['revisits']} |")
    if note:
        row += f" {note} |"
    return row


HEADER = ("| scene | N | bbox x/y/z span (m) | med step (m) | med rot (deg) | "
          "med rot/step (deg/m) | revisits |\n"
          "|---|---|---|---|---|---|---|")

HEADER_GROUP = ("| scene | N | bbox x/y/z span (m) | med step (m) | med rot (deg) | "
                "med rot/step (deg/m) | revisits | group |\n"
                "|---|---|---|---|---|---|---|---|")


def build_table(title, scenes, loader, notes=None):
    notes = notes or {}
    lines = [f"### {title}", "", HEADER_GROUP if notes else HEADER]
    rows = []
    for scene in scenes:
        try:
            c2w, n_bad = loader(scene)
            s = scene_stats(c2w)
            note = notes.get(scene, "")
            if n_bad:
                note = (note + f" ({n_bad} unreadable pose files skipped)").strip()
            lines.append(fmt_row(scene, s, note))
            rows.append((scene, s))
        except Exception as e:
            err = f"| {scene} | FAILED | - | - | - | - | - | {type(e).__name__}: {e} |"
            lines.append(err + " |" if notes else err)
            traceback.print_exc()
    return lines, rows


def main():
    lines = [
        "# Scene GT Trajectory Statistics",
        "",
        "GT poses as read by the benchmark loaders "
        "(`src/depth_anything_3/bench/datasets/sevenscenes.py`, "
        "`src/depth_anything_3/bench/datasets/scannetpp.py`).",
        "- 7scenes: `frame-XXXXXX.pose.txt` = 4x4 c2w, all `seq-*` frames on disk "
        "(one seq per scene: seq-01, stairs seq-02), sorted by frame index.",
        "- scannetpp: COLMAP `merge_dslr_iphone/colmap/sparse_render_rgb`, w2c -> c2w; "
        "iPhone frames only (the evaluated trajectory; dslr/render_rgb excluded), "
        "sorted by frame index.",
        f"- revisit = frame pair with center distance < {REVISIT_DIST_M} m and "
        f"frame-index gap > {REVISIT_MIN_GAP}.",
        "- med rot/step = median of per-adjacent-pair rotation/translation ratios "
        "(deg/m); pairs with step <= 1e-6 m excluded from the median.",
        "",
        "Generated by `diagnostics/free_geometry/scene_traj_stats.py` (read-only).",
        "",
    ]

    t7, rows7 = build_table("7scenes", SEVENSCENES_SCENES, load_7scenes_c2w)
    notes = {s: "wound" for s in SCANNETPP_WOUND}
    notes.update({s: "healthy" for s in SCANNETPP_HEALTHY})
    tsn, rowssn = build_table("scannetpp", SCANNETPP_WOUND + SCANNETPP_HEALTHY,
                              load_scannetpp_c2w, notes)

    lines += t7 + ["", ""] + tsn + [""]
    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {OUT_MD}")

    # Quick group summaries for the handoff
    def summarize(rows, label):
        if not rows:
            return f"{label}: no rows"
        r = {k: np.median([s[k] for _, s in rows]) for k in
             ("med_step", "med_rot", "med_ratio", "revisits")}
        bbox = np.median([s["bbox"] for _, s in rows], axis=0)
        return (f"{label}: med bbox=({bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f}) "
                f"step={r['med_step']:.4f} rot={r['med_rot']:.2f} "
                f"ratio={r['med_ratio']:.0f} revisits={r['revisits']:.0f}")

    print("\n" + summarize(rows7, "7scenes all"))
    print(summarize([(s, v) for s, v in rows7 if s == "office"], "7scenes office"))
    print(summarize([(s, v) for s, v in rows7 if s != "office"], "7scenes others"))
    print(summarize([(s, v) for s, v in rowssn if s in SCANNETPP_WOUND], "scannetpp wound"))
    print(summarize([(s, v) for s, v in rowssn if s in SCANNETPP_HEALTHY], "scannetpp healthy"))


if __name__ == "__main__":
    main()
