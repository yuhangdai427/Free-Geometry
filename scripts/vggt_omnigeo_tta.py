#!/usr/bin/env python3
"""VGGT + our TTA (rkdc1h: maskdistill + 1.5x RKD-Huber + couple) on OmniGeo,
evaluated EXACTLY with Self-Evolve's benchmark protocol (eval branch):

- dataset: OmniGeoDataset layout (<seq>/image/*.png + camera_poses.npy C2W +
  intrinsics.npy), 49 sequences at
  /root/autodl-tmp/OmniWorld/benchmark/geometric_prediction
- preprocessing: SelfEvo `load_and_preprocess_images` mode="crop" (width 518,
  aspect-scaled, center-cropped; [0,1] tensors, NO ImageNet norm — verbatim)
- frame sampling: sparse 1/10 (endpoints kept), min 2 frames
- metric: all-pairs relative-pose rotation & translation-direction angle
  errors + Racc/Tacc/AUC@{5,15,30}deg via SelfEvo relpose/metric.py (imported)

TTA per sequence (our unified protocol, adapted to 518):
- teacher = frozen VGGT-1B; student = LoRA r32/a32 on all 24 aggregator
  layers (VGGT's aggregator IS the multi-view part; per-view DinoV2 stays
  frozen), heads frozen
- 10 training pairs/seq: 8 random teacher frames (seeded), shared student
  slots [0,2,4,6]; 100 steps AdamW lr3e-5 wd1e-5, warmup15%+cosine, clip 1.0
- loss = maskdistill(all positions, teacher-conf weighted) + 1.5*RKD-Huber
  (delta 0.2) + 1.0*couple; input corruption = 50% 14x14 image-block masking
- image_hw=(518,518) for all pose decodes
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/autodl-tmp/Free-Geometry"
SELFEVO_RELPOSE = "/root/autodl-tmp/SelfEvo/relpose"
OMNIGEO_DIR = "/root/autodl-tmp/OmniWorld/benchmark/geometric_prediction"
OMNIVIDEO_DIR = "/root/autodl-tmp/OmniWorld/benchmark/video_generation"

sys.path.insert(0, os.path.join(ROOT, "src"))              # our vggt (wins)
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))
sys.path.append(SELFEVO_RELPOSE)                            # SelfEvo metric

import common  # noqa: E402  (diagnostics/free_geometry/common.py)
import modeling as M  # noqa: E402
from common import STUDENT_INDICES, TAP_LAYERS, stable_seed  # noqa: E402
from train_arms import (  # noqa: E402
    loss_b5_maskdistill, mask_image_blocks)
from abs_pose_loss import loss_rkd_shared_pose_huber, loss_couple  # noqa: E402

from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

# SelfEvo's metric module (verbatim protocol): all-pairs R/t angle + AUC
from metric import se3_to_relative_pose_error, calculate_auc_np  # noqa: E402
sys.path.append("/root/autodl-tmp/SelfEvo/videodepth")
from depth import depth_evaluation  # noqa: E402  (gamegeo scale&shift protocol)
import cv2  # noqa: E402

IMAGE_HW = (518, 518)
PATCH_HW = (37, 37)          # 518 / 14
THRESHOLDS = [5, 15, 30]     # SelfEvo configs/data/relpose-angular.yaml: OmniGeo
LR, WD, CLIP, STEPS, WARMUP_RATIO = 3e-5, 1e-5, 1.0, 100, 0.15


# ---------------------------------------------------------------- SelfEvo data
def _try_int_stem(fname):
    try:
        return int(os.path.splitext(fname)[0])
    except ValueError:
        return 10 ** 18


def list_sequences(root=OMNIGEO_DIR):
    seqs = []
    for d in sorted(os.listdir(root)):
        seq_dir = os.path.join(root, d)
        image_dir = os.path.join(seq_dir, "image")
        cam = os.path.join(seq_dir, "camera_poses.npy")
        intr = os.path.join(seq_dir, "intrinsics.npy")
        if not (os.path.isdir(image_dir) and os.path.isfile(cam) and os.path.isfile(intr)):
            continue
        files = [f for f in os.listdir(image_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=_try_int_stem)
        n = min(len(files), np.load(cam, mmap_mode="r").shape[0],
                np.load(intr, mmap_mode="r").shape[0])
        if n >= 2:
            seqs.append({"name": d, "dir": seq_dir,
                         "files": [os.path.join(image_dir, f) for f in files][:n],
                         "n": n})
    return seqs


def load_images_selfevo(paths):
    """VERBATIM replication of SelfEvo eval_angle.load_and_preprocess_images
    (mode='crop'): width->518 aspect (multiple of 14), center-crop height to
    518, [0,1] tensors, no normalization."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    target = 518
    images, shapes = [], set()
    for p in paths:
        img = Image.open(p)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        w, h = img.size
        new_w = target
        new_h = max(14, round(h * (new_w / w) / 14) * 14)
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        ten = TF.to_tensor(img)
        if new_h > target:  # center crop
            start_y = (new_h - target) // 2
            ten = ten[:, start_y:start_y + target, :]
        shapes.add((ten.shape[1], ten.shape[2]))
        images.append(ten)
    assert len(shapes) == 1, f"shape mismatch after crop: {shapes}"
    return torch.stack(images)  # (S,3,518,518) in [0,1]


def sparse_ids(n, divisor=10, min_frames=2):
    """SelfEvo maybe_sparse_sample_ids (sparse=True): ~n/divisor uniform
    frames incl. endpoints."""
    target = max(min_frames, n // max(1, divisor))
    target = min(target, n)
    picked = np.linspace(0, n - 1, target, dtype=int)
    return np.unique(picked)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_sequence(student, seq, device="cuda"):
    ids = sparse_ids(seq["n"])
    paths = [seq["files"][i] for i in ids]
    imgs = load_images_selfevo(paths).to(device)
    feats24, psi, preds = M.student_preds(student, imgs.unsqueeze(0))
    pose_enc = preds["pose_enc"]
    with torch.autocast(device_type="cuda", enabled=False):
        ext, _ = pose_encoding_to_extri_intri(pose_enc.float().cpu(), IMAGE_HW)
    ext = ext[0]  # (S,3,4) w2c
    S = ext.shape[0]
    last = torch.tensor([0, 0, 0, 1]).view(1, 4).expand(S, 1, 4)
    pred44 = torch.cat([ext, last], dim=-2)  # (S,4,4) w2c

    c2w = np.load(os.path.join(seq["dir"], "camera_poses.npy"))[:seq["n"]][ids]
    gt44 = torch.linalg.inv(torch.from_numpy(c2w).float())  # w2c

    rel_r, rel_t = se3_to_relative_pose_error(
        pred_se3=pred44, gt_se3=gt44, num_frames=len(ids))
    row = {"seq": seq["name"], "n_eval_frames": int(len(ids))}
    for th in THRESHOLDS:
        row[f"Racc_{th}"] = float((rel_r < th).float().mean()) * 100
        row[f"Tacc_{th}"] = float((rel_t < th).float().mean()) * 100
        auc, _ = calculate_auc_np(rel_r.numpy(), rel_t.numpy(), max_threshold=th)
        row[f"Auc_{th}"] = float(auc) * 100
    row["r_err_mean"] = float(rel_r.mean())
    row["t_err_mean"] = float(rel_t.mean())

    # ---- video depth (SelfEvo gamegeo: same sparse ids, SAME forward) ----
    # pred (S,h,w) -> cv2.resize CUBIC to GT size; per-sequence scale&shift
    # (align_with_lad2, median-init, 1000 iters); mask = gt>0; max_depth None.
    depth_dir = os.path.join(seq["dir"], "depth")
    dfiles = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
    assert len(dfiles) == seq["n"], f"depth/GT count mismatch {seq['name']}"
    gt = np.stack([np.load(os.path.join(depth_dir, dfiles[int(i)]))
                   for i in ids])
    pr_full = preds["depth"].float().cpu().numpy()[0]
    pr = np.stack([cv2.resize(pr_full[k], (gt.shape[2], gt.shape[1]),
                              interpolation=cv2.INTER_CUBIC)
                   for k in range(pr_full.shape[0])])
    # SelfEvo's align_with_lad2 runs its own 1000-iter optimization with
    # .backward(); the enclosing @torch.no_grad() would break it -> re-enable
    with torch.enable_grad():
        dres, _, _, _ = depth_evaluation(pr, gt, max_depth=None, use_gpu=True,
                                         align_with_lad2=True)
    for k, v in dres.items():
        row[f"depth_{k}"] = float(v)

    del imgs, feats24, preds
    torch.cuda.empty_cache()
    return row, rel_r, rel_t


# ---------------------------------------------------------------- TTA (ours)
def train_sequence(teacher, student, seq, device="cuda", seed=0,
                   on_step=None, log=print, steps=STEPS, lr=LR):
    """One-sequence TTA, our unified protocol at 518."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    n = seq["n"]
    rng = np.random.RandomState(
        stable_seed("omnigeo_pairs", seq["name"], seed) % (2 ** 32))

    # 10 training pairs: 8 seeded-random teacher frames each
    pairs = []
    for _ in range(10):
        t = sorted(rng.choice(n, size=8, replace=False).tolist())
        pairs.append(t)

    # teacher caches (one 8-view forward per pair); patch grid is DYNAMIC —
    # OmniGeo frames are wide (e.g. 294x518), SelfEvo crop mode keeps aspect
    caches, images4_all = [], []
    patch_hw = None
    for t in pairs:
        imgs8 = load_images_selfevo([seq["files"][i] for i in t]) \
            .unsqueeze(0).to(device)
        ph, pw = imgs8.shape[-2] // 14, imgs8.shape[-1] // 14
        if patch_hw is None:
            patch_hw = (ph, pw)
        assert (ph, pw) == patch_hw, f"intra-seq shape drift {ph},{pw}"
        caches.append(M.cache_teacher_pair(teacher, imgs8, patch_hw))
        images4_all.append(imgs8[0, STUDENT_INDICES].cpu())
        del imgs8
    torch.cuda.empty_cache()

    base = M.get_base_vggt(student)
    torch.manual_seed(stable_seed("lora_init", seq["name"], seed))
    M.reset_lora_(student)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    warm = max(1, round(steps * WARMUP_RATIO))
    scheduler = SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warm),
         CosineAnnealingLR(optimizer, T_max=steps - warm, eta_min=1e-8)],
        [warm])

    student.train()
    step = 0
    n_epochs = (steps + 9) // 10
    ones = torch.ones(1, 4, patch_hw[0] * patch_hw[1], device=device)
    for epoch in range(n_epochs):
        order = [int(i) for i in torch.randperm(
            10, generator=torch.Generator().manual_seed(
                stable_seed("train_order", seq["name"], epoch, seed)))]
        for pi in order:
            if step >= steps:
                break
            cache = caches[pi]
            images4 = images4_all[pi].unsqueeze(0).to(device)
            gen = torch.Generator(device=device).manual_seed(
                stable_seed("mask", seq["name"], epoch, pi, seed))
            images4_in, _ = mask_image_blocks(images4, 0.5, patch_hw, gen)
            feats24, psi, preds = M.student_preds(student, images4_in)
            pt = cache["pose_enc8"][:, STUDENT_INDICES].float()
            feat, _ = loss_b5_maskdistill(
                base.depth_head, cache, feats24, patch_hw, ones)
            rkd, _ = loss_rkd_shared_pose_huber(
                preds["pose_enc"], pt, image_hw=IMAGE_HW)
            cp, _ = loss_couple(
                preds["pose_enc"], preds["depth"], pt,
                cache["depth4"], cache.get("conf4"), image_hw=IMAGE_HW)
            loss = feat + 1.5 * rkd + 1.0 * cp
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss {float(loss)} at {seq['name']} step {step+1}")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if on_step is not None:
                on_step({"step": step, "pair_idx": pi, "epoch": epoch,
                         "loss": float(loss), "maskdistill": float(feat),
                         "rkd": float(rkd), "couple": float(cp),
                         "grad_norm": float(gn),
                         "lr": scheduler.get_last_lr()[0]})
            del images4, images4_in, feats24, preds
        if step >= steps:
            break
    student.eval()
    return step


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n shards by sequence index")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="default: workspace/{benchmark}_vggt_rkdc[_s{steps}]")
    ap.add_argument("--skip_tta", action="store_true", help="baseline eval only")
    ap.add_argument("--swanlab", action="store_true")
    ap.add_argument("--benchmark", default="omnigeo",
                    choices=["omnigeo", "omnivideo"],
                    help="omnigeo = benchmark/geometric_prediction (49 seqs); "
                         "omnivideo = benchmark/video_generation (187 seqs x 81 frames)")
    ap.add_argument("--steps", type=int, default=100,
                    help="TTA steps per sequence (try 1000 for the "
                         "longer-training hypothesis; epochs auto-adjust)")
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    bench_dir = OMNIGEO_DIR if args.benchmark == "omnigeo" else OMNIVIDEO_DIR

    if args.out is None:
        args.out = f"workspace/{args.benchmark}_vggt_rkdc"
        if args.steps != 100:
            args.out += f"_s{args.steps}"
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    seqs = list_sequences(bench_dir)
    i, nshard = (int(x) for x in args.shard.split("/"))
    seqs = [s for k, s in enumerate(seqs) if k % nshard == i]
    print(f"[{args.benchmark}] total sequences on disk: "
          f"{len(list_sequences(bench_dir))}; "
          f"this shard ({args.shard}): {len(seqs)}", flush=True)

    run = None
    if args.swanlab:
        import swanlab
        run = swanlab.init(project=f"free-geometry-{args.benchmark.replace('omni', 'omni-')}",
                           experiment_name=f"vggt_rkdc_s{args.steps}_shard{args.shard}",
                           config={"arm": "rkdc1h", "steps": args.steps, "lr": args.lr,
                                   "image_hw": list(IMAGE_HW),
                                   "protocol": "SelfEvo eval-branch relpose-angular"})

    teacher = M.load_teacher(device)
    student = M.load_student(device)

    results_path = os.path.join(
        args.out, f"results_shard{args.shard.replace('/', '_')}.json")
    rows_path = results_path.replace(".json", ".csv")
    all_rows = []
    for seq in seqs:
        t0 = time.time()
        # ---- baseline (zero-LoRA == frozen VGGT) ----
        M.reset_lora_(student)
        student.eval()
        base_row, _, _ = eval_sequence(student, seq, device)

        # ---- our TTA ----
        if not args.skip_tta:
            trace = []
            nsteps = train_sequence(
                teacher, student, seq, device, seed=args.seed,
                on_step=(lambda r: trace.append(r)) if run is not None else None,
                steps=args.steps, lr=args.lr)
            tta_row, _, _ = eval_sequence(student, seq, device)
        else:
            nsteps, tta_row, trace = 0, None, []

        row = {"seq": seq["name"], "n_frames": seq["n"]}
        keys = [k for k in base_row if k != "seq"]
        for k in keys:
            row[f"base_{k}"] = base_row[k]
            if tta_row is not None:
                row[f"tta_{k}"] = tta_row[k]
                row[f"d_{k}"] = tta_row[k] - base_row[k]
        row["train_steps"] = nsteps
        row["time_s"] = time.time() - t0
        all_rows.append(row)
        with open(results_path, "w") as f:
            json.dump(all_rows, f, indent=1)
        keys = list(row.keys())
        if not os.path.exists(rows_path):
            with open(rows_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                f.flush()
                os.fsync(f.fileno())
        with open(rows_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writerow(row)
        if run is not None and tta_row is not None:
            run.log({f"{seq['name']}/auc15_base": base_row["Auc_15"],
                     f"{seq['name']}/auc15_tta": tta_row["Auc_15"],
                     f"{seq['name']}/d_auc15": tta_row["Auc_15"] - base_row["Auc_15"]})
        dm = base_row.get("depth_Abs Rel", float("nan"))
        dm_t = (tta_row or {}).get("depth_Abs Rel", float("nan"))
        print(f"[{seq['name']}] n={seq['n']} base AUC15={base_row['Auc_15']:.2f}"
              f" absrel={dm:.4f}"
              + (f" | tta AUC15={tta_row['Auc_15']:.2f} "
                 f"({tta_row['Auc_15']-base_row['Auc_15']:+.2f}) "
                 f"absrel={dm_t:.4f} ({dm_t-dm:+.4f})" if tta_row else "")
              + f" [{row['time_s']:.0f}s]", flush=True)

    # ---- overall ----
    def overall(prefix):
        out = {}
        for k in ("Auc_5", "Auc_15", "Auc_30", "Racc_30", "Tacc_30",
                  "depth_Abs Rel", "depth_Sq Rel", "depth_RMSE",
                  "depth_Log RMSE", "depth_δ < 1.25", "depth_δ < 1.25^2",
                  "depth_δ < 1.25^3"):
            vals = [r[f"{prefix}{k}"] for r in all_rows if f"{prefix}{k}" in r]
            if vals:
                out[k] = float(np.mean(vals))
        return out

    summary = {"shard": args.shard, "n_seq": len(all_rows),
               "base": overall("base_"),
               "tta": overall("tta_") if not args.skip_tta else None,
               "per_sequence": all_rows}
    with open(results_path.replace(".json", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sequence"},
                     indent=1))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
