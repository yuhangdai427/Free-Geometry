#!/usr/bin/env python3
"""Run the four locked 0.1-Co-visibility VGGT comparison arms.

The output is deliberately sequence-addressed.  Metrics are first averaged over
the three fixed sequences of a scene and only then over scenes, so a scene with
more available views cannot receive extra weight.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "vggt")]
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.bench.datasets.eth3d import ETH3D
from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
from depth_anything_3.bench.datasets.sevenscenes import SevenScenes
from depth_anything_3.bench.datasets.hiroom import HiRoomDataset
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.test_time_adaption.covisibility_sequences import load_selected_scenes, sequence_calibration
from vggt.vggt.test_time_adaption.models import VGGTStudentModel

ARMS = ("base_8v", "base_4v", "base_8v_extract_4v", "lora_4v", "lora_8v", "base_16v", "lora_16v")
DEFAULT_ARMS = ("base_8v", "base_4v", "base_8v_extract_4v", "lora_4v", "lora_8v")


def load_images(paths, image_size):
    images = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        scale = image_size / max(h, w)
        new_h, new_w = max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        images.append(image.astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(0)


def with_extrinsics(predictions, images):
    pose = predictions["pose_enc"]
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    _, _, _, h, w = images.shape
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, image_size_hw=(h, w), pose_encoding_type="absT_quaR_FoV")
    predictions["extrinsics"] = extrinsics.squeeze(0)
    predictions["intrinsics"] = intrinsics.squeeze(0)
    return predictions


@torch.no_grad()
def infer_direct(model, images):
    return with_extrinsics(model(images), images)


@torch.no_grad()
def infer_8v_extract_4v(model, images8):
    """Run the 8V aggregator, then decode tokens exactly at 0,2,4,6."""
    tokens8, patch_start = model.aggregator(images8)
    positions = [0, 2, 4, 6]
    tokens4 = [tokens[:, positions].contiguous() for tokens in tokens8]
    images4 = images8[:, positions].contiguous()
    for full, sliced in zip(tokens8, tokens4):
        if not torch.equal(sliced, full[:, positions]):
            raise AssertionError("8V-to-4V token slice was altered")
    predictions = {}
    with torch.cuda.amp.autocast(enabled=False):
        if model.camera_head is not None:
            predictions["pose_enc"] = model.camera_head(tokens4)[-1]
        if model.depth_head is not None:
            predictions["depth"], predictions["depth_conf"] = model.depth_head(tokens4, images=images4, patch_start_idx=patch_start)
        if model.point_head is not None:
            predictions["world_points"], predictions["world_points_conf"] = model.point_head(tokens4, images=images4, patch_start_idx=patch_start)
    return with_extrinsics(predictions, images4)


def save_prediction(path, predictions, image_files, extrinsics, intrinsics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output = {"gt_extrinsics": extrinsics, "gt_intrinsics": intrinsics}
    for key, value in predictions.items():
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
            if value.ndim and value.shape[0] == 1:
                value = value[0]
            output[key] = value
    np.savez_compressed(path, **output)
    # Match VGGTEvaluator's on-disk contract so dataset fusion uses exactly
    # these sampled frames rather than loading a full scene.
    np.savez_compressed(
        os.path.join(Path(path).parent.parent, "gt_meta.npz"),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_files=np.asarray(image_files, dtype=str),
    )


def pose_metrics(predictions, gt_extrinsics):
    metrics = compute_pose(torch.from_numpy(as_homogeneous(predictions["extrinsics"].cpu().numpy())), torch.from_numpy(as_homogeneous(gt_extrinsics)))
    return {key: float(value) for key, value in metrics.items()}


def scannetpp_with_real_rgb(dataset, scene_name):
    """Repair only this evaluator instance's RGB paths for TSDF fusion.

    ScanNet++ COLMAP names are ``iphone/<frame>.jpg`` while its legacy loader
    joins them below ``images/`` and skips every frame.  This local repair keeps
    the benchmark loader untouched and lets its existing undistort/fusion code
    operate on its intended data.
    """
    original_get_data = dataset.get_data

    def get_data(scene):
        input_root = os.path.join(dataset.data_root, scene, "merge_dslr_iphone")
        had_cache = scene in dataset._scene_cache
        if had_cache:
            del dataset._scene_cache[scene]
        # Temporarily present image paths in the form expected by the legacy loader.
        image_root = os.path.abspath(os.path.join(input_root, "images"))
        original_exists = os.path.exists
        os.path.exists = lambda path: original_exists(path) or original_exists(
            os.path.join(image_root, "iphone", os.path.basename(path))
        ) if path.startswith(image_root + os.sep) else original_exists(path)
        try:
            data = original_get_data(scene)
        finally:
            os.path.exists = original_exists
        data.image_files = [os.path.join(image_root, "iphone", os.path.basename(path)) for path in data.image_files]
        dataset._scene_cache[scene] = data
        return data

    dataset.get_data = get_data
    dataset.get_data(scene_name)
    return dataset


def hiroom_with_absolute_rgb(dataset, scene_name):
    """Normalize the legacy relative HiRoom paths for sampled TSDF fusion."""
    data = dataset.get_data(scene_name)
    data.image_files = [os.path.abspath(path) for path in data.image_files]
    dataset._scene_cache[scene_name] = data
    return dataset


def means(items):
    keys = set().union(*(item.keys() for item in items))
    return {key: float(np.mean([item[key] for item in items if key in item])) for key in sorted(keys)}


def write_metrics(output_root, all_metrics):
    """Persist valid incremental JSON before any bulky arm payload is removed."""
    summary, window_metrics = {}, {}
    for (dataset, arm), by_scene in all_metrics.items():
        scene_means = {scene: means(values) for scene, values in by_scene.items()}
        summary.setdefault(dataset, {})[arm] = {"scenes": scene_means, "dataset_mean": means(list(scene_means.values()))}
        window_metrics.setdefault(dataset, {})[arm] = by_scene
    Path(output_root).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_root, "pose_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(output_root, "window_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(window_metrics, handle, indent=2, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--selection", default="artifacts/covisibility_all_datasets_0025/nonoverlap_010_counts.jsonl")
    parser.add_argument("--datasets", nargs="+", default=["eth3d", "scannetpp"])
    parser.add_argument("--model-name", default="model_weights/VGGT-1B")
    parser.add_argument("--lora-eth3d")
    parser.add_argument("--lora-scannetpp")
    parser.add_argument("--lora-7scenes")
    parser.add_argument("--lora-hiroom")
    parser.add_argument("--output-root", default="results/covisibility_010_vggt")
    parser.add_argument("--image-size", type=int, default=504)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--lora-layers-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recon-unposed", action="store_true", help="Compute TSDF reconstruction F1 for every sequence.")
    parser.add_argument("--delete-arm-results", action="store_true", help="Delete an arm's predictions after its metrics JSON is safely written.")
    # 16V uses a distinct manifest and is invoked explicitly by the launcher.
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS))
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.sequence_manifest, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    for record in records:
        frame_ids = record.get("eight_frame_ids", [])
        if len(frame_ids) != 8 or len(set(frame_ids)) != 8:
            raise ValueError(
                f"{record.get('dataset')}/{record.get('scene')}: strict 8V evaluation "
                "requires exactly eight unique frames; refusing a padded manifest"
            )
        if record.get("four_frame_ids") != [frame_ids[index] for index in (0, 2, 4, 6)]:
            raise ValueError(
                f"{record.get('dataset')}/{record.get('scene')}: 4V must be the 0/2/4/6 subset of 8V"
            )
    scenes = {(scene.dataset, scene.scene): scene for scene in load_selected_scenes(args.selection, args.datasets)}
    records = [record for record in records if record["dataset"] in args.datasets]
    base = VGGT.from_pretrained(args.model_name).to(device).eval()
    lora_paths = {"eth3d": args.lora_eth3d, "scannetpp": args.lora_scannetpp,
                  "7scenes": args.lora_7scenes, "hiroom": args.lora_hiroom}
    lora_models = {}
    all_metrics = defaultdict(lambda: defaultdict(list))
    # Complete one arm at a time. This bounds disk use to one arm, and lets us
    # delete prediction/TSDF artifacts immediately after durable JSON output.
    for arm in args.arms:
        for record in records:
            scene = scenes[(record["dataset"], record["scene"])]
            if arm in {"lora_4v", "lora_8v", "lora_16v"}:
                path = lora_paths[record["dataset"]]
                if not path:
                    raise ValueError(f"--lora-{record['dataset']} is required for {arm}")
                if record["dataset"] not in lora_models:
                    student = VGGTStudentModel(model_name=args.model_name, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_layers=list(range(args.lora_layers_start, 24)))
                    student.load_lora_weights(path)
                    lora_models[record["dataset"]] = student.to(device).eval()._get_vggt_model()
                model = lora_models[record["dataset"]]
            else:
                model = base
            image_files, gt_ext, gt_int = sequence_calibration(scene, record, arm)
            if arm == "base_8v_extract_4v":
                model_images = record["eight_image_files"]
            elif arm in {"base_16v", "lora_16v"}:
                model_images = record["sixteen_image_files"]
            else:
                model_images = image_files
            images = load_images(model_images, args.image_size).to(device)
            if arm == "base_8v_extract_4v":
                predictions = infer_8v_extract_4v(model, images)
            else:
                predictions = infer_direct(model, images)
            path = os.path.join(args.output_root, record["dataset"], arm, f"sample_{record['sample_idx']:03d}", record["scene"], "exports", "mini_npz", "results.npz")
            save_prediction(path, predictions, image_files, gt_ext, gt_int)
            metrics = pose_metrics(predictions, gt_ext)
            if args.recon_unposed:
                if record["dataset"] == "eth3d":
                    dataset = ETH3D()
                elif record["dataset"] == "7scenes":
                    dataset = SevenScenes()
                elif record["dataset"] == "hiroom":
                    dataset = hiroom_with_absolute_rgb(HiRoomDataset(), record["scene"])
                elif record["dataset"] == "scannetpp":
                    dataset = scannetpp_with_real_rgb(ScanNetPP(), record["scene"])
                else:
                    raise ValueError(f"Unsupported reconstruction dataset: {record['dataset']}")
                fuse_path = os.path.join(Path(path).parents[2], "fuse", "pcd.ply")
                dataset.fuse3d(record["scene"], path, fuse_path, "recon_unposed")
                metrics.update({key: float(value) for key, value in dataset.eval3d(record["scene"], fuse_path).items()})
            all_metrics[(record["dataset"], arm)][record["scene"]].append(metrics)
        summary = write_metrics(args.output_root, all_metrics)
        if args.delete_arm_results:
            for dataset in args.datasets:
                payload = Path(args.output_root) / dataset / arm
                if payload.is_dir():
                    shutil.rmtree(payload)
                    print(f"[pruned arm] {payload}", flush=True)
    # A reconstruction-only rerun may begin from an existing pose JSON.  Merge
    # its historical arms so appending F1/CD never discards already-computed
    # AUC metrics for arms not selected on this invocation.
    metrics_path = Path(args.output_root) / "pose_metrics.json"
    window_path = Path(args.output_root) / "window_metrics.json"
    if metrics_path.exists() and window_path.exists():
        previous_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        previous_windows = json.loads(window_path.read_text(encoding="utf-8"))
        for dataset, arms in previous_windows.items():
            for arm, scenes_for_arm in arms.items():
                if (dataset, arm) not in all_metrics:
                    all_metrics[(dataset, arm)] = defaultdict(list, scenes_for_arm)
    summary = write_metrics(args.output_root, all_metrics)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
