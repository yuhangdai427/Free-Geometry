"""Frozen/adapted exports share frame IDs and the standard benchmark metric chain."""

import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, model_identity
from .sampling import validate_manifest
from .trainer import write_json


def export_scene(adapter, manifest, config, output_dir, checkpoint=None):
    validate_manifest(manifest)
    if manifest["status"] != "ready":
        raise ValueError(manifest["reason"])
    root = Path(output_dir)
    path = root / "exports" / "mini_npz" / "results.npz"
    if path.exists():
        raise FileExistsError(path)
    images, valid, metadata = adapter.prepare_images(manifest["image_files"])
    adapter.amp_enabled = config.train.amp
    identity = model_identity(config, adapter.resolve_weights())
    adapter.load(config.train.device)
    step = 0
    if checkpoint:
        payload = load_checkpoint(checkpoint, config, manifest, identity)
        adapter.teacher.to("cpu")
        adapter.reset_student(config.train.device)
        adapter.load_trainable_state(payload["parameters"])
        step = payload["step"]
    indices = manifest["eval_frames"]
    with torch.no_grad():
        out = adapter.export_eval(images[indices][None].to(config.train.device))
    depth = np.asarray(out["depth"], np.float32)
    conf = np.asarray(out["conf"], np.float32)
    conf = np.where(valid[indices].numpy(), conf, 0)
    ext = np.asarray(out["extr_w2c"], np.float32)
    intr = np.asarray(out["intr"], np.float32)
    if ext.shape[-2:] == (3, 4):
        bottom = np.broadcast_to(np.array([0, 0, 0, 1], np.float32), (len(ext), 1, 4))
        ext = np.concatenate((ext, bottom), 1)
    if not all(np.isfinite(v).all() for v in (depth, conf, ext, intr)):
        raise FloatingPointError("evaluation produced nonfinite predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        depth=depth,
        conf=conf,
        extrinsics=ext,
        intrinsics=intr,
        frame_ids=np.asarray([manifest["frame_ids"][i] for i in indices]),
        valid=valid[indices].numpy(),
    )
    write_json(root / "manifest.json", manifest)
    write_json(root / "preprocessing.json", metadata)
    write_json(
        root / "export.json",
        {
            "model": config.model.name,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "step": step,
            "frame_ids": [manifest["frame_ids"][i] for i in indices],
            "pose_diagnostics": getattr(adapter, "pose_diagnostics", None),
        },
    )
    return path


def benchmark_export(
    manifest, output_dir, data_root, gt_root=None, reconstruction=True
):
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous

    from .datasets import metric_dataset

    dataset = metric_dataset(manifest["dataset"], data_root, gt_root)
    scene = manifest["scene"]
    gt = dataset.get_data(scene)
    by_path = {str(Path(p).resolve()): i for i, p in enumerate(gt.image_files)}
    files = [manifest["image_files"][i] for i in manifest["eval_frames"]]
    missing = [p for p in files if str(Path(p).resolve()) not in by_path]
    if missing:
        raise ValueError(f"evaluation GT missing for selected frames: {missing}")
    indices = [by_path[str(Path(p).resolve())] for p in files]
    root = Path(output_dir)
    pred_path = root / "exports" / "mini_npz" / "results.npz"
    metadata = json.loads((root / "preprocessing.json").read_text())
    with np.load(pred_path) as z:
        arrays = benchmark_arrays(
            dict(z), [metadata[i] for i in manifest["eval_frames"]]
        )
    # Existing benchmark consumers assume an uncropped image canvas.
    pred_path = root / "exports" / "benchmark_npz" / "results.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **arrays)
    ext, pred_depth, pixel_valid = (
        arrays["extrinsics"],
        arrays["depth"],
        arrays["valid"],
    )
    gt_ext = np.asarray(gt.extrinsics)[indices]
    pose = compute_pose(
        as_homogeneous(torch.from_numpy(ext).float()),
        as_homogeneous(torch.from_numpy(gt_ext).float()),
    )
    metrics = {
        key: float(getattr(pose, key)) for key in ("auc03", "auc05", "auc15", "auc30")
    }
    metrics.update(
        depth_metrics(manifest, pred_depth, files, gt, indices, data_root, pixel_valid)
    )
    # Official TSDF/reconstruction consumers expect GT metadata for precisely these images.
    payload = {
        "extrinsics": gt_ext,
        "intrinsics": np.asarray(gt.intrinsics)[indices],
        "image_files": np.asarray(files, dtype=object),
    }
    aux = gt.aux
    if aux.get("mask_files") is not None:
        payload["mask_files"] = np.asarray(
            [aux["mask_files"][i] for i in indices], dtype=object
        )
    np.savez_compressed(root / "exports" / "gt_meta.npz", **payload)
    if reconstruction and manifest["dataset"] != "dtu64":
        fused = root / "exports" / "fuse" / "pcd.ply"
        fused.parent.mkdir(parents=True, exist_ok=True)
        dataset.fuse3d(scene, str(pred_path), str(fused), "recon_unposed")
        metrics.update(
            {
                "recon_" + k: float(v)
                for k, v in dict(dataset.eval3d(scene, str(fused))).items()
            }
        )
    write_json(root / "metrics.json", metrics)
    return metrics


def depth_metrics(
    manifest, pred_depth, files, dataset_data, indices, data_root, pixel_valid
):
    """Historical scene-wide least-squares depth alignment, GT used only here."""
    import cv2
    from PIL import Image

    name = manifest["dataset"]
    if name in ("dtu", "dtu64"):
        return {"depth_metric_status": "not_available"}
    gt_files = dataset_data.aux.get("gt_depth_files")
    depths = []
    for j, path in enumerate(files):
        if name == "eth3d":
            relative = Path(path).relative_to(
                Path(data_root).resolve() / manifest["scene"] / "images"
            )
            depth_path = (
                Path(data_root) / manifest["scene"] / "ground_truth_depth" / relative
            )
            with Image.open(path) as im:
                w, h = im.size
            d = np.fromfile(depth_path, dtype=np.float32).reshape(h, w)
        else:
            if not isinstance(gt_files, (list, tuple)) or not gt_files:
                raise ValueError("benchmark dataset does not provide GT depth paths")
            d = cv2.imread(str(gt_files[indices[j]]), cv2.IMREAD_UNCHANGED)
            if d is None:
                raise FileNotFoundError(gt_files[indices[j]])
            d = d.astype(np.float32) * (100.0 / 65535 if name == "hiroom" else 0.001)
        depths.append(
            cv2.resize(
                d,
                (pred_depth.shape[-1], pred_depth.shape[-2]),
                interpolation=cv2.INTER_NEAREST,
            )
        )
    gt = np.stack(depths)
    limit = {"eth3d": 150.0, "hiroom": 100.0, "7scenes": 10.0, "scannetpp": 5.0}[name]
    valid = (
        pixel_valid
        & np.isfinite(gt)
        & (gt > 0)
        & (gt <= limit)
        & np.isfinite(pred_depth)
        & (pred_depth > 0)
    )
    if not valid.any():
        raise ValueError("depth evaluation has no valid support")
    p, g = pred_depth[valid].astype(np.float64), gt[valid].astype(np.float64)
    scale = (p * g).sum() / max((p * p).sum(), 1e-12)
    aligned = p * scale
    return {
        "depth_metric_status": "ok",
        "depth_valid_pixels": int(valid.sum()),
        "depth_ls_scale": float(scale),
        "abs_rel": float((np.abs(aligned - g) / g).mean()),
        "delta125": float((np.maximum(aligned / g, g / aligned) < 1.25).mean()),
    }


def benchmark_arrays(native, metadata):
    """Undo crop/padding into a common full-FOV canvas before legacy metrics.

    Per-frame resizing is allowed by the standard benchmark readers. Intrinsics
    must undergo the same inverse transform as depth, including crop offsets.
    """
    import cv2

    h, w = native["depth"].shape[-2:]
    output = {key: value.copy() for key, value in native.items()}
    for i, item in enumerate(metadata):
        oh, ow = item["original_hw"]
        target_scale = np.diag([w / ow, h / oh, 1.0])
        transform = target_scale @ np.linalg.inv(np.asarray(item["image_transform"]))
        output["intrinsics"][i] = transform @ native["intrinsics"][i]
        support = cv2.warpAffine(
            native["valid"][i].astype(np.uint8),
            transform[:2],
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        output["valid"][i] = support
        for field in ("depth", "conf"):
            value = cv2.warpAffine(
                native[field][i],
                transform[:2],
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            output[field][i] = np.where(support, value, 0)
    return output
