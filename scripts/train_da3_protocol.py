#!/usr/bin/env python3
"""DA3-giant-1.1 per-scene TTA — Free-Geometry protocol v1 (C2M: maskdistill + rel-pose).

Port of the verified VGGT protocol (docs/TTA_PROTOCOL_v1_2026-09-13.md) to DA3.
Zero-GT frame selection (N/tau rules), frozen teacher cache, LoRA r32/a32 on all
40 aggregator blocks, AdamW 3e-5 cosine, fixed 100 steps, then pose/depth eval on
the protocol eval frames (benchmark-100 for N>=100, else allv).

Smoke example (2 ScanNet++ scenes):
    python scripts/train_da3_protocol.py \
        --scenes 09c1414f1b 1ada7a0617 \
        --output_root workspace/da3_protocol_smoke \
        --steps 100
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))

import common as fg_common
from depth_anything_3.test_time_adaption import protocol_v1 as P


def get_scene_data(scene: str):
    """Dispatch via diagnostics.free_geometry.common (set_dataset first): identical
    image_lists as the frozen manifests for all 4 datasets."""
    return fg_common.get_scene_data(scene)


def load_gt_depth_raw(path: str, out_hw) -> np.ndarray:
    """GT depth via the shared 4-dataset convention (common.load_gt_depth)."""
    return fg_common.load_gt_depth(path, out_hw)


def cv2_imread_unchanged(path):
    import cv2

    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def cv2_resize(d, wh):
    import cv2

    return cv2.resize(d, wh, interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def evaluate_scene(student, scene_data, eval_frames, max_frames: int = 0,
                   scene: str = "", dataset_obj=None, export_dir: str = None) -> dict:
    """ONE inference on the protocol eval frames with the in-memory student (LoRA
    active), producing ALL metrics together: pose AUC (compute_pose, w2c),
    scale-fitted AbsRel / δ1.25, AND recon_unposed F1/CD (mini_npz export ->
    gt_meta -> fuse3d TSDF -> eval3d) when dataset_obj+export_dir are given."""
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous

    frames = list(eval_frames)
    if max_frames > 0 and len(frames) > max_frames:
        frames = frames[:max_frames]
    image_files = [scene_data.image_files[i] for i in frames]

    student.eval()
    pred = student.da3.inference(
        image=image_files,
        process_res=P.PROCESS_RES,
        process_res_method="upper_bound_resize",
        ref_view_strategy="first",
        export_dir=export_dir,
        export_format="mini_npz" if export_dir else "mini_npz",
    )
    depth = np.asarray(pred.depth, dtype=np.float32)  # [N,H,W]
    ext = np.asarray(pred.extrinsics, dtype=np.float32)  # [N,4,4] w2c

    gt_ext = np.asarray(scene_data.extrinsics)[frames]  # w2c
    pose = compute_pose(
        as_homogeneous(torch.from_numpy(ext).float()),
        as_homogeneous(torch.from_numpy(gt_ext).float()),
    )

    # Depth metrics with one least-squares scale per scene (GT-valid px only).
    gt = np.stack([
        load_gt_depth_raw(scene_data.aux.gt_depth_files[i], depth.shape[-2:]) for i in frames
    ])
    omega = np.isfinite(gt) & (gt > 0) & np.isfinite(depth) & (depth > 0)
    n_valid = int(omega.sum())
    out = {
        "n_eval_frames": len(frames),
        "auc03": float(pose.auc03),
        "auc05": float(pose.auc05),
        "auc15": float(pose.auc15),
        "auc30": float(pose.auc30),
        "n_valid_px": n_valid,
    }
    if n_valid > 0:
        p, g = depth[omega].astype(np.float64), gt[omega].astype(np.float64)
        scale = float((p * g).sum() / max((p * p).sum(), 1e-12))
        ps = p * scale
        abs_rel = float(np.mean(np.abs(ps - g) / g))
        delta125 = float(np.mean(np.maximum(ps / g, g / ps) < 1.25))
        abs_rel_raw = float(np.mean(np.abs(p - g) / g))
        out.update({
            "abs_rel": abs_rel,
            "delta125": delta125,
            "ls_scale": scale,
            "abs_rel_unscaled": abs_rel_raw,
        })

    # ---- recon_unposed F1/CD from the SAME inference's npz export ----
    if dataset_obj is not None and export_dir is not None:
        import zipfile

        result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
        deadline = time.time() + 60
        while True:
            try:
                with zipfile.ZipFile(result_path) as z:
                    if z.testzip() is None:
                        break
            except (FileNotFoundError, EOFError, OSError, zipfile.BadZipFile):
                pass
            if time.time() > deadline:
                raise TimeoutError(f"export timeout: {result_path}")
            time.sleep(0.1)
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        payload = {
            "extrinsics": np.asarray(scene_data.extrinsics)[frames],
            "intrinsics": np.asarray(scene_data.intrinsics)[frames],
            "image_files": np.array(image_files, dtype=object),
        }
        aux = scene_data.aux
        if getattr(aux, "get", None) and aux.get("mask_files") is not None:
            payload["mask_files"] = np.array([aux["mask_files"][i] for i in frames], dtype=object)
        np.savez_compressed(meta_path, **payload)
        fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        dataset_obj.fuse3d(scene, result_path, fuse_path, "recon_unposed")
        recon = dataset_obj.eval3d(scene, fuse_path)
        for k, v in dict(recon).items():
            out[f"recon_{k}"] = float(v)
    return out


def make_dataset(ds):
    import importlib

    table = {
        "scannetpp": ("depth_anything_3.bench.datasets.scannetpp", "ScanNetPP"),
        "7scenes": ("depth_anything_3.bench.datasets.sevenscenes", "SevenScenes"),
        "hiroom": ("depth_anything_3.bench.datasets.hiroom", "HiRoomDataset"),
        "eth3d": ("depth_anything_3.bench.datasets.eth3d", "ETH3D"),
    }
    mod = importlib.import_module(table[ds][0])
    return getattr(mod, table[ds][1])()


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3 per-scene TTA (protocol v1, C2M)")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default="scannetpp")
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    ap.add_argument("--output_root", default="workspace/da3_protocol_smoke")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=P.EPOCHS)
    ap.add_argument("--lr", type=float, default=P.LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose_weight", type=float, default=1.0,
                    help="1.0 = C2M (default); 0.0 = pure maskdistill (protocol fine "
                         "variant for tau<=0.55 datasets)")
    ap.add_argument("--arm", default="c2m", choices=["c2m", "pw0", "rkdc1h", "rkdc1hc"],
                    help="c2m = maskdistill+rel (maskrel); pw0 = pure maskdistill; "
                         "rkdc1h = maskdistill + 1.5*rkd_huber + 1.0*couple (no rel); "
                         "rkdc1hc = rkdc1h + ctk_weight*camera-token KD")
    ap.add_argument("--ratio_mix", default=None,
                    help="comma-separated teacher:student ratios mixed within ONE run, "
                         "e.g. '8:4,16:4,24:8' (SelfEvo-style mixed asymmetry per pair)")
    ap.add_argument("--selfevo", action="store_true",
                    help="SelfEvo-faithful student sampling: teacher-window FIRST+LAST "
                         "anchored, per-pair random L middle frames")
    ap.add_argument("--combo", action="store_true",
                    help="combined sampling: fixed-slot pairs (camera) + "
                         "endpoint-anchored L>=4 pairs (fusion), see --combo_se_frac")
    ap.add_argument("--combo_se_frac", type=float, default=0.5,
                    help="fraction of endpoint-anchored pairs in combo mode "
                         "(0.5 = 1:1, 0.25 = 3:1 fixed:SE)")
    ap.add_argument("--ctk_weight", type=float, default=1.0,
                    help="weight of the camera-token KD term (rkdc1hc arm)")
    ap.add_argument("--two_stage", type=float, default=0.0,
                    help=">0 (e.g. 0.7): sequential curriculum — fixed-slot pairs for "
                         "the first frac*steps, then endpoint-anchored pairs (combo "
                         "mode only; disables early_stop)")
    ap.add_argument("--mask_ratio", type=float, default=0.5,
                    help="input patch mask ratio for the distill loss; 0.0 = nomask "
                         "(clean-input distillation at all positions)")
    ap.add_argument("--n_shared", type=int, default=4,
                    help="shared (student) frames per pair: 4 = 16:4 protocol, 8 = 16:8 "
                         "(unsaturates the student's local targets on strong backbones)")
    ap.add_argument("--n_train", type=int, default=10,
                    help="train pairs per scene (protocol default 10; use 5 for tiny-N "
                         "scenes to avoid pair-overlap overfitting)")
    ap.add_argument("--teacher_N", type=int, default=0,
                    help="teacher window override (0 = protocol default 16/8): "
                         "e.g. 8 for 8:4/8:2, 32 for 32:8 ratio variants")
    ap.add_argument("--eval_max_frames", type=int, default=0,
                    help="cap eval frames (0 = protocol frames as-is)")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--early_stop", action="store_true",
                    help="stop training when tail-10 loss mean improves <2%% over "
                         "prev-10 (min 30 steps) — avoids over-training on saturated scenes")
    args = ap.parse_args()

    device = "cuda"
    fg_common.set_dataset(args.dataset)
    os.makedirs(args.output_root, exist_ok=True)
    trace_path = os.path.join(args.output_root, "training_trace.csv")
    summary_path = os.path.join(args.output_root, "smoke_summary.json")
    dataset_obj = make_dataset(args.dataset)

    print(f"Loading teacher + student: {args.model_name}")
    teacher = P.create_teacher(args.model_name)  # CPU; moved to GPU per phase
    student = P.create_student(args.model_name)

    trace_rows: list = []
    summary = {"model": args.model_name, "args": vars(args), "scenes": {}}

    for scene in args.scenes:
        t0 = time.time()
        scene_data = get_scene_data(scene)
        image_files = list(scene_data.image_files)
        ratio_mix = None
        if args.ratio_mix:
            ratio_mix = []
            for tok in args.ratio_mix.split(","):
                t_, s_ = tok.strip().split(":")
                ratio_mix.append((int(t_), int(s_)))
        proto = P.build_scene_protocol(image_files, scene, dataset=args.dataset,
                                       n_train=args.n_train,
                                       n_shared=args.n_shared,
                                       teacher_N=(args.teacher_N or None),
                                       ratio_mix=ratio_mix,
                                       selfevo=args.selfevo,
                                       combo=args.combo,
                                       combo_se_frac=args.combo_se_frac)
        print(f"[{scene}] N={proto['N']} tau={proto['tau']:.3f} "
              f"teacher_N={proto['teacher_N']} strategy={proto['strategy']} "
              f"n_shared={proto['n_shared']} eval_frames={len(proto['eval_frames'])}")

        pose_w = 0.0 if args.arm == "pw0" else args.pose_weight
        stats = P.train_scene_c2m(
            teacher, student, scene, image_files, proto,
            device=device, steps=args.steps, epochs=args.epochs,
            lr=args.lr, seed=args.seed, pose_weight=pose_w,
            arm=args.arm, mask_ratio=args.mask_ratio, trace_rows=trace_rows,
            ctk_weight=args.ctk_weight, two_stage=args.two_stage,
            early_stop=args.early_stop)

        ckpt_dir = os.path.join(args.output_root, "ckpts", scene)
        os.makedirs(ckpt_dir, exist_ok=True)
        student.save_lora_weights(os.path.join(ckpt_dir, "c2m_final_lora.pt"))

        summary["scenes"][scene] = {
            "protocol": {k: proto[k] for k in ("N", "tau", "teacher_N", "strategy")},
            "n_eval_frames": len(proto["eval_frames"]),
            "train": stats,
            "train_time_s": time.time() - t0,
        }
        # Keep the protocol on disk for reproducibility (frame lists included).
        with open(os.path.join(ckpt_dir, "protocol.json"), "w") as f:
            json.dump(proto, f, indent=2)

        if not args.skip_eval:
            # Eval with the just-trained adapter still active in memory (avoids
            # a PEFT same-name adapter reload). Teacher stays resident but idle.
            # ONE inference -> AUC + AbsRel + F1/CD out together (no backfill).
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            ev = evaluate_scene(student, scene_data, proto["eval_frames"],
                                max_frames=args.eval_max_frames,
                                scene=scene, dataset_obj=dataset_obj,
                                export_dir=os.path.join(args.output_root, "recon", scene))
            ev["eval_peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
            ev["eval_time_s"] = time.time() - t0
            summary["scenes"][scene]["eval"] = ev
            print(f"[{scene}] eval: auc03={ev['auc03']:.4f} "
                  f"fscore={ev.get('recon_fscore', float('nan')):.4f} "
                  f"cd={ev.get('recon_overall', float('nan')):.4f} "
                  f"abs_rel={ev.get('abs_rel', float('nan')):.4f} "
                  f"({ev['n_eval_frames']} frames, peak {ev['eval_peak_mem_mib']:.0f}MiB)")

    del teacher
    torch.cuda.empty_cache()

    if trace_rows:
        keys = ["scene", "step", "epoch", "pair_idx", "loss", "lr", "grad_norm",
                "peak_mem_mib", "maskdistill", "rel", "rel_rot", "rel_tdir",
                "rkd_sh_d", "rkd_sh_a", "couple", "ctk", "mask_ratio"]
        with open(trace_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(trace_rows)
        print(f"Trace written to {trace_path}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")
    print("DONE")


if __name__ == "__main__":
    main()
