#!/usr/bin/env python3
"""Longer-is-better probe on scannetpp train pairs (2026-09-23), v2.

Per scene x 10 manifest train pairs x {DA3, VGGT}:
  - CLEAN teacher forwards on growing prefixes k in {4,8,16} (nested contexts)
  - one forward on the 4 student frames with the campaign 50% block mask
Metrics per pass (all vs GT): pose AUC@3, depth abs_rel (LS-scaled, GT-valid),
and recon_unposed F1 via TSDF fuse3d + eval3d. The heavy CPU fusion/eval runs
in an 8-worker process pool pipelined behind the GPU forward passes.

Usage: python scripts/probe_longer.py <da3|vggt>
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Thread caps MUST precede numpy/o3d imports: spawned pool workers re-execute
# this module top-down, so these land before their BLAS/OpenMP initializes.
# (Setting them in the pool initializer was too late — load-average ~1000.)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "src", "vggt"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

import common as fg
from common import load_gt_depth, load_image_model
from depth_anything_3.test_time_adaption import protocol_v1 as P

MAN = "workspace/ndispatch/scannetpp/scene_manifest.json"
OUTDIR = "workspace/overnight/longer_probe"
KS = [4, 8, 16]
FUSE_WORKERS = 4  # each worker caps its BLAS/OpenMP threads (see _pool_init):
                 # 4 workers x 4 threads stays inside the shared-host CPU budget.

_pool_ds = None


def _pool_init(dataset="scannetpp"):
    # Thread caps MUST be set before numpy/o3d import in this spawned worker;
    # uncapped Open3D/OpenMP was the load-average-900 oversubscription cause.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "4"
    global _pool_ds
    fg.set_dataset(dataset)
    if dataset == "eth3d":
        from depth_anything_3.bench.datasets.eth3d import ETH3D
        _pool_ds = ETH3D()
    else:
        from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
        _pool_ds = ScanNetPP()


def fuse_eval_task(scene, export_dir):
    """Runs in a pool worker: TSDF fuse + eval3d -> recon_unposed fscore."""
    export_dir = os.path.abspath(export_dir)
    result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
    os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
    try:
        _pool_ds.fuse3d(scene, result_path, fuse_path, "recon_unposed")
        recon = _pool_ds.eval3d(scene, fuse_path)
        return {f"recon_{k}": float(v) for k, v in recon.items()
                if isinstance(v, (int, float))}
    except Exception as exc:
        return {"recon_error": str(exc)}


def pose_auc3(ext_w2c, gt_w2c):
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous
    pose = compute_pose(as_homogeneous(torch.from_numpy(np.asarray(ext_w2c)).float()),
                        as_homogeneous(torch.from_numpy(np.asarray(gt_w2c)).float()))
    return float(pose.auc03)


def depth_absrel(depth, gt_depth_files, frames, hw):
    g = np.stack([load_gt_depth(gt_depth_files[i], hw) for i in frames])
    d = np.asarray(depth, dtype=np.float64)
    omega = np.isfinite(g) & (g > 0) & np.isfinite(d) & (d > 0)
    if omega.sum() == 0:
        return None
    p, gg = d[omega], g[omega]
    s = float((p * gg).sum() / max((p * p).sum(), 1e-12))
    return float(np.mean(np.abs(p * s - gg) / gg))


def export_cell(export_dir, ext, intr, depth, image_files, frames, aux):
    """mini_npz results + gt_meta (same layout the bench fuse3d/eval3d expect)."""
    e = os.path.join(export_dir, "exports")
    os.makedirs(os.path.join(e, "mini_npz"), exist_ok=True)
    np.savez_compressed(os.path.join(e, "mini_npz", "results.npz"),
                        depth=np.asarray(depth, dtype=np.float32),
                        extrinsics=np.asarray(ext, dtype=np.float32),
                        intrinsics=np.asarray(intr, dtype=np.float32))
    payload = {
        "extrinsics": np.asarray(aux["gt_ext"], dtype=np.float32)[frames],
        "intrinsics": np.asarray(aux["gt_intr"], dtype=np.float32)[frames],
        "image_files": np.array([image_files[f] for f in frames], dtype=object),
    }
    mf = aux.get("mask_files")
    if mf is not None:
        payload["mask_files"] = np.array([mf[f] for f in frames], dtype=object)
    np.savez_compressed(os.path.join(e, "gt_meta.npz"), **payload)


class DA3Wrap:
    def __init__(self):
        self.student = P.create_student("model_weights/DA3-GIANT-1.1")
        self.student.to("cuda").eval()
        self.model = self.student.da3 if hasattr(self.student, "da3") else self.student

    def forward(self, image_files, mask_seed=None, gt_intr=None):
        if mask_seed is None:
            with torch.no_grad():
                pred = self.model.inference(image=image_files, process_res=P.PROCESS_RES,
                                            process_res_method="upper_bound_resize",
                                            ref_view_strategy="first")
            ext = np.asarray(pred.extrinsics, dtype=np.float32)
            intr = np.asarray(pred.intrinsics, dtype=np.float32)
            depth = np.asarray(pred.depth, dtype=np.float32)
            return ext, intr, depth, (depth.shape[-2], depth.shape[-1])
        # Masked student forward: the SAME path training uses
        # (load_images_da3 -> mask_image_blocks with the CUDA generator seeded
        # by stable_seed("mask", scene, 0, pi, 0) -> student_forward_c2m).
        images4 = P.load_images_da3(image_files).to("cuda").unsqueeze(0)
        ph, pw = images4.shape[-2] // 14, images4.shape[-1] // 14
        gen = torch.Generator(device="cuda").manual_seed(mask_seed)
        images4_in, _ = P.mask_image_blocks(images4, 0.5, (ph, pw), gen)
        with torch.no_grad():
            _, ext_w2c, depth_s = P.student_forward_c2m(self.student, images4_in,
                                                        with_depth=True)
        depth = depth_s.squeeze(0).squeeze(-1).float().cpu().numpy()
        ext = ext_w2c[0].float().cpu().numpy()
        # fuse3d reads GT intrinsics from gt_meta; predicted intr only fills
        # the results.npz schema for the unposed reconstruction path.
        intr = np.asarray(gt_intr, dtype=np.float32)
        return ext, intr, depth, (depth.shape[-2], depth.shape[-1])


class VGGTWrap:
    def __init__(self):
        import modeling as M
        self.M = M
        self.student = M.load_student()
        self.student.to("cuda").eval()

    def forward(self, image_files, mask_seed=None, gt_intr=None):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        base = self.M.get_base_vggt(self.student)
        imgs = [load_image_model(f) for f in image_files]
        arr = torch.from_numpy(np.stack(imgs, 0)).permute(0, 3, 1, 2).float().unsqueeze(0)
        if mask_seed:
            ph, pw = arr.shape[-2] // 14, arr.shape[-1] // 14
            gen = torch.Generator(device="cuda").manual_seed(mask_seed)
            arr, _ = P.mask_image_blocks(arr.to("cuda"), 0.5, (ph, pw), gen)
            images = arr
        else:
            images = arr.to("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            preds = base(images)
        depth = preds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()
        pose_enc = preds["pose_enc"]
        if pose_enc.dim() == 2:
            pose_enc = pose_enc.unsqueeze(0)
        ext, intr = pose_encoding_to_extri_intri(pose_enc, image_size_hw=images.shape[-2:],
                                                 pose_encoding_type="absT_quaR_FoV")
        return (ext.squeeze(0).float().cpu().numpy(), intr.squeeze(0).float().cpu().numpy(),
                depth, (depth.shape[-2], depth.shape[-1]))




def dump_merged(results, dst):
    """Concurrent-writer safe dump: re-read, merge, write."""
    try:
        disk = json.load(open(dst))
        disk.update(results)
        results = disk
    except Exception:
        pass
    json.dump(results, open(dst, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["da3", "vggt"])
    ap.add_argument("--dataset", default="scannetpp", choices=["scannetpp", "eth3d"])
    ap.add_argument("--ckpt_root", default=None,
                    help="per-scene TTA checkpoint root; presence switches the "
                         "probe to post-adaptation mode (loads each scene's LoRA)")
    ap.add_argument("--drain_only", action="store_true",
                    help="skip forwards; recompute recon metrics for cells that "
                         "already have exports on disk (F1 backfill)")
    args = ap.parse_args()

    tag = "_adapted" if args.ckpt_root else ""
    suffix = "" if args.dataset == "scannetpp" else f"_{args.dataset}"
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    dst = os.path.join(OUTDIR, f"longer_probe_{args.model}{tag}{suffix}.json")
    results = json.load(open(dst)) if os.path.exists(dst) else {}

    fg.set_dataset(args.dataset)
    man = json.load(open(os.path.join("workspace", "ndispatch", args.dataset,
                                      "scene_manifest.json")))
    wrap = DA3Wrap() if args.model == "da3" else VGGTWrap()
    import multiprocessing
    pool = ProcessPoolExecutor(max_workers=FUSE_WORKERS, initializer=_pool_init,
                               initargs=(args.dataset,),
                               mp_context=multiprocessing.get_context("spawn"))

    if args.drain_only:
        futures = []
        for scene, pairs in sorted(results.items()):
            for pi, row in pairs.items():
                for cell, m in row.items():
                    if "recon_fscore" in m or "recon_error" in m:
                        continue
                    edir = os.path.join(OUTDIR, "exports", args.model, scene,
                                        f"p{pi}_{cell}")
                    if not os.path.isdir(edir):
                        m["recon_error"] = "exports missing"
                        continue
                    futures.append((scene, pi, cell,
                                    pool.submit(fuse_eval_task, scene, edir)))
        for scene, pi, cell, fut in futures:
            try:
                results[scene][pi][cell].update(fut.result(timeout=1800))
            except Exception as exc:
                results[scene][pi][cell]["recon_error"] = str(exc)
        dump_merged(results, dst)
        pool.shutdown()
        print(f"DRAIN-ONLY DONE: {len(futures)} cells -> {dst}")
        return
    futures = []  # (scene, pi, cell, future)

    for scene, sc in sorted(man["scenes"].items()):
        if scene in results:
            continue
        only = os.environ.get("PROBE_ONLY")
        if only and scene not in only.split(","):
            continue
        limit = os.environ.get("PROBE_LIMIT")
        if limit and len(results) >= int(limit):
            break
        if args.ckpt_root:
            if args.model == "da3":
                ck = os.path.join(args.ckpt_root, scene, "ckpts", scene, "c2m_final_lora.pt")
            else:
                ck = os.path.join(args.ckpt_root, "ckpts", scene, "C2M_rawrel",
                                  "step100_lora_peft")
                if not os.path.isdir(ck):
                    ck = ck.replace("step100_lora_peft", "step100_lora.pt")
            wrap.student.load_lora_weights(ck)
            print(f"[{scene}] loaded TTA ckpt: {ck}", flush=True)
        sd = fg.get_scene_data(scene)
        files = list(sd.image_files)
        aux = {
            "gt_ext": np.asarray(sd.extrinsics),
            "gt_intr": np.asarray(sd.intrinsics),
            "mask_files": (sd.aux.get("mask_files")
                           if sd.aux.get("mask_files") is not None else None),
        }
        gtf = sd.aux.get("gt_depth_files")
        t0 = time.time()
        rows = {}
        for pi, pair in enumerate(sc["train_pairs"]):
            tf, sf = pair["teacher_frames"], pair["student_frames"]
            slot = [tf.index(f) for f in sf]  # student frames' slots inside the teacher context
            row = {}
            tn = sc.get("teacher_N", len(tf))
            tcell = f"teacher_k{tn}"
            # One teacher forward serves two cells: the heads decode per-frame,
            # so slicing the student slots out of the 16-frame output IS
            # "decode only the student frames from the teacher-context tokens".
            edir16 = os.path.join(OUTDIR, "exports", args.model, scene, f"p{pi}_{tcell}")
            ext, intr, depth, hw = wrap.forward([files[f] for f in tf])
            row[tcell] = {
                "auc03": pose_auc3(ext, aux["gt_ext"][tf]),
                "abs_rel": depth_absrel(depth, gtf, tf, hw) if gtf else None,
            }
            export_cell(edir16, ext, intr, depth, files, tf, aux)
            futures.append((scene, str(pi), tcell,
                            pool.submit(fuse_eval_task, scene, edir16)))
            # Distillation-target cell: teacher-context outputs AT the student
            # slots — exactly what the student is distilled toward in training.
            edir_sl = os.path.join(OUTDIR, "exports", args.model, scene, f"p{pi}_teacher_slot4")
            ext_sl, intr_sl, depth_sl = ext[slot], intr[slot], depth[slot]
            row["teacher_slot4"] = {
                "auc03": pose_auc3(ext_sl, aux["gt_ext"][sf]),
                "abs_rel": depth_absrel(depth_sl, gtf, sf, hw) if gtf else None,
            }
            export_cell(edir_sl, ext_sl, intr_sl, depth_sl, files, sf, aux)
            futures.append((scene, str(pi), "teacher_slot4",
                            pool.submit(fuse_eval_task, scene, edir_sl)))
            for cell, frames, mask_seed in [
                ("student_clean", sf, None),
                ("student_masked", sf, P.stable_seed("mask", scene, 0, pi, 0)),
            ]:
                edir = os.path.join(OUTDIR, "exports", args.model, scene, f"p{pi}_{cell}")
                ext, intr, depth, hw = wrap.forward(
                    [files[f] for f in frames], mask_seed,
                    gt_intr=aux["gt_intr"][frames] if mask_seed else None)
                row[cell] = {
                    "auc03": pose_auc3(ext, aux["gt_ext"][frames]),
                    "abs_rel": depth_absrel(depth, gtf, frames, hw) if gtf else None,
                }
                export_cell(edir, ext, intr, depth, files, frames, aux)
                futures.append((scene, str(pi), cell,
                                pool.submit(fuse_eval_task, scene, edir)))
            rows[str(pi)] = row
        results[scene] = rows
        dump_merged(results, dst)
        print(f"[{scene}] 10 pairs forwarded in {time.time()-t0:.0f}s "
              f"(fuse/eval pipelined, backlog={len(futures)})", flush=True)

    print("forwards done; draining fuse/eval pool ...", flush=True)
    for scene, pi, cell, fut in futures:
        try:
            results[scene][pi][cell].update(fut.result(timeout=1200))
        except Exception as exc:
            results[scene][pi][cell]["recon_error"] = str(exc)
        dump_merged(results, dst)
    pool.shutdown()
    print("ALL DONE ->", dst)


if __name__ == "__main__":
    main()
