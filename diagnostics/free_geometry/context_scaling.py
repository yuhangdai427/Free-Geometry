#!/usr/bin/env python3
"""Context-scaling diagnostic: does a longer teacher context (16/32 views)
reduce the ~58% of patches where the 8->4 teacher's local depth is worse than
the frozen 4-view student?

Per scene x ctx in {8,16,32} x train pair (10/scene), pure forward, no training:
  a. teacher cache at ctx views (cache_teacher_pair_ctx), frozen 4-view student
     forward on the shared views (same frozen backbone, student input path);
  b. per-patch centered squared log residuals vs GT (perpatch_logres2);
  c. stats: % patches teacher-worse, mean residuals, share of per-patch
     |log(student)-log(teacher)| difference mass carried by teacher-worse patches;
  d. pose AUC (auc03/auc30) of teacher pose_encN sliced at shared_slots, plus
     the 4-view student baseline, vs GT extrinsics.

Outputs: artifacts/diagnostics/context_scaling/results.csv (scene x ctx x pair)
and summary.md (means by ctx). No images/npz are saved.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import get_scene_data, load_image_model, load_manifest
import modeling as M
from train_arms import perpatch_logres2

OUT_DIR_DEFAULT = "artifacts/diagnostics/context_scaling"
CTX_MANIFEST_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "manifest.json")
REFERENCE_PCT_WORSE = 0.5715  # bakeoff_v1 patch_audit: 1 - mean frac_patches_teacher_better

FIELDS = [
    "scene", "pair_id", "ctx", "n_views",
    "pct_teacher_worse", "mean_pp_teacher", "mean_pp_student",
    "diffmass_share_teacher_worse",
    "teacher_auc03", "teacher_auc30", "student_auc03", "student_auc30",
    "peak_mem_gb",
]


def load_images_n(scene_data, frames, device="cuda"):
    imgs = [load_image_model(scene_data.image_files[i]) for i in frames]
    arr = np.stack(imgs, axis=0)  # [N,H,W,3]
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0)
    return t.to(device)


def perpatch_abs_logdiff(d_t, d_s, gt4, patch_hw):
    """Per-patch mean |log d_t - log d_s| over GT-valid pixels -> [4,P]."""
    ph, pw = patch_hw
    valid = np.isfinite(gt4) & (gt4 > 0)
    d = np.abs(np.log(np.clip(d_t, 1e-6, None)) - np.log(np.clip(d_s, 1e-6, None)))
    d = np.where(valid, d, 0.0)
    S, H, W = d.shape
    return d.reshape(S, ph, H // ph, pw, W // pw).mean(axis=(2, 4)).reshape(S, ph * pw)


@torch.no_grad()
def compute_one(vggt, scene_data, scene, pair_id, ctx, frames, slots, shared_frames):
    torch.cuda.reset_peak_memory_stats()
    imagesN = load_images_n(scene_data, frames)
    H, W = imagesN.shape[-2:]
    patch_hw = (H // 14, W // 14)

    cache = M.cache_teacher_pair_ctx(vggt, imagesN, slots, patch_hw)

    images4 = imagesN[:, slots].contiguous()
    feats4, psi = M.aggregator_all(vggt, images4)
    depth_s, _ = M.replay_depth_nograd(vggt, feats4, images4, psi)
    pose_s = M.replay_camera_nograd(vggt, feats4)

    gt4 = M.load_probe_gt(scene_data, shared_frames, (H, W))
    d_t = cache["depth4"].squeeze(0).squeeze(-1).cpu().numpy()
    d_s = depth_s.squeeze(0).squeeze(-1).float().cpu().numpy()

    pp_t = perpatch_logres2(d_t, gt4, patch_hw)
    pp_s = perpatch_logres2(d_s, gt4, patch_hw)
    worse = pp_t > pp_s
    w = perpatch_abs_logdiff(d_t, d_s, gt4, patch_hw)
    mass = float(w[worse].sum() / max(w.sum(), 1e-12))

    gt_ext = np.asarray(scene_data.extrinsics)[shared_frames]
    auc_t = M.pose_auc(cache["pose_encN"][:, slots], gt_ext)
    auc_s = M.pose_auc(pose_s.float(), gt_ext)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del cache, feats4, depth_s, pose_s, imagesN, images4
    torch.cuda.empty_cache()
    return {
        "scene": scene, "pair_id": pair_id, "ctx": ctx, "n_views": len(frames),
        "pct_teacher_worse": float(worse.mean()),
        "mean_pp_teacher": float(pp_t.mean()),
        "mean_pp_student": float(pp_s.mean()),
        "diffmass_share_teacher_worse": mass,
        "teacher_auc03": auc_t["auc03"], "teacher_auc30": auc_t["auc30"],
        "student_auc03": auc_s["auc03"], "student_auc30": auc_s["auc30"],
        "peak_mem_gb": round(peak, 2),
    }


def write_summary(rows, out_path):
    ctxs = sorted({r["ctx"] for r in rows})
    metrics = ["pct_teacher_worse", "diffmass_share_teacher_worse",
               "mean_pp_teacher", "mean_pp_student",
               "teacher_auc03", "teacher_auc30", "student_auc03", "student_auc30",
               "peak_mem_gb"]
    lines = ["# Context scaling: teacher local depth quality vs context length", "",
             f"Reference (bakeoff_v1 patch_audit, ctx=8, probe pairs): "
             f"%teacher-worse = {REFERENCE_PCT_WORSE:.4f}", "",
             "| ctx | n_rows | " + " | ".join(metrics) + " |",
             "|---" * (len(metrics) + 2) + "|"]
    for c in ctxs:
        rs = [r for r in rows if r["ctx"] == c]
        means = [float(np.mean([r[m] for r in rs])) for m in metrics]
        lines.append(f"| {c} | {len(rs)} | "
                     + " | ".join(f"{m:.4f}" for m in means) + " |")
    lines += ["",
              "pct_teacher_worse: fraction of patches where the teacher's per-patch "
              "centered squared log residual vs GT exceeds the frozen student's.",
              "diffmass_share_teacher_worse: share of per-patch |log(t)-log(s)| mass "
              "carried by teacher-worse patches.",
              "teacher/student_aucXX: pose AUC on the shared 4 views (teacher pose_encN "
              "sliced at shared_slots; student = frozen 4-view forward).",
              "peak_mem_gb: max over rows of torch.cuda.max_memory_allocated per pair.", "",
              "## Reproduction note (tie semantics)", "",
              "The 0.5715 reference counts TIES (pp_t == pp_s) as teacher-worse "
              "(patch_audit: better = pp_s - pp_t > 0). Ties are fully GT-invalid "
              "14x14 patches (both residuals 0): 14.2% of patches on probe pairs, "
              "11.8% on train pairs, and are context-independent. Reproduction at ctx=8:",
              "- probe pairs (original split): strict 0.4299, ties-as-worse 0.5715 (exact match)",
              "- train pairs (this experiment): strict 0.4079, ties-as-worse 0.5258",
              "ties-as-worse for ctx 16/32 = strict + 0.1178 (mean train-pair tie frac):",
              "~0.494 (ctx16), ~0.474 (ctx32).", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_manifest", default="artifacts/diagnostics/bakeoff_v1/scene_manifest.json")
    ap.add_argument("--ctx_manifest", default=CTX_MANIFEST_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--ctxs", nargs="*", type=int, default=[8, 16, 32])
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--oom_retries", type=int, default=3)
    args = ap.parse_args()

    import json
    man = load_manifest(args.scene_manifest)
    with open(args.ctx_manifest, "r", encoding="utf-8") as f:
        ctxman = json.load(f)
    scenes = args.scenes or sorted(man["scenes"])

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "results.csv")
    write_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    fcsv = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()

    vggt = M.load_teacher()
    rows = []
    for scene in scenes:
        scene_data = get_scene_data(scene)
        pairs = ctxman["scenes"][scene]["train_pairs"]
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        for pair in pairs:
            for ctx in args.ctxs:
                c = pair["ctx"][str(ctx)]
                frames, slots = c["frames"], c["shared_slots"]
                row = None
                for attempt in range(args.oom_retries + 1):
                    try:
                        row = compute_one(vggt, scene_data, scene, pair["pair_id"],
                                          ctx, frames, slots, pair["shared_frames"])
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if attempt == args.oom_retries:
                            raise
                        print(f"[{scene} p{pair['pair_id']} ctx{ctx}] OOM, retry in 60s "
                              f"({attempt + 1}/{args.oom_retries})", flush=True)
                        time.sleep(60)
                writer.writerow(row)
                fcsv.flush()
                rows.append(row)
                print(f"[{scene} p{pair['pair_id']} ctx{ctx:>2}] "
                      f"worse={row['pct_teacher_worse']:.4f} "
                      f"mass={row['diffmass_share_teacher_worse']:.4f} "
                      f"tAUC3={row['teacher_auc03']:.3f} sAUC3={row['student_auc03']:.3f} "
                      f"peak={row['peak_mem_gb']:.1f}GB", flush=True)
        del scene_data
        torch.cuda.empty_cache()
    fcsv.close()

    print(write_summary(rows, os.path.join(args.out_dir, "summary.md")), flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
