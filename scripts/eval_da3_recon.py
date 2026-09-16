#!/usr/bin/env python3
"""DA3 per-scene F1/CD (recon_unposed) evaluation on frozen protocol manifests.

Per scene: inference on the protocol eval frames (base model OR the scene's own
LoRA ckpt hot-swapped into one resident StudentModel), export mini_npz, write
gt_meta.npz (subset GT), then the bench dataset's fuse3d (TSDF) + eval3d
({acc, comp, overall, precision, recall, fscore}).

Examples:
  # baseline arm (no LoRA)
  python scripts/eval_da3_recon.py --dataset 7scenes \
      --manifest artifacts/diagnostics/final_protocol/7scenes/scene_manifest.json \
      --arm A0_baseline --out artifacts/diagnostics/final_protocol/da3_baseline/7scenes_recon_baseline.json
  # TTA arm with per-scene ckpts
  python scripts/eval_da3_recon.py --dataset 7scenes \
      --manifest artifacts/diagnostics/final_protocol/7scenes/scene_manifest.json \
      --arm rkdc1h --ckpt_root workspace/da3_protocol_7scenes_rkdc1h/ckpts \
      --out artifacts/diagnostics/final_protocol/da3_baseline/7scenes_recon_rkdc1h.json
"""

import argparse
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

DATASET_CLASSES = {
    "scannetpp": ("depth_anything_3.bench.datasets.scannetpp", "ScanNetPP"),
    "7scenes": ("depth_anything_3.bench.datasets.sevenscenes", "SevenScenes"),
    "hiroom": ("depth_anything_3.bench.datasets.hiroom", "HiRoomDataset"),
    "eth3d": ("depth_anything_3.bench.datasets.eth3d", "ETH3D"),
}


def make_dataset(ds):
    import importlib

    mod = importlib.import_module(DATASET_CLASSES[ds][0])
    return getattr(mod, DATASET_CLASSES[ds][1])()


def load_scene_lora_(student, ckpt_dir):
    """Hot-swap the scene's adapter weights + camera token into the resident
    student (avoids PEFT same-name adapter reload issues)."""
    from safetensors.torch import load_file

    peft_dir = os.path.join(ckpt_dir, "c2m_final_lora_peft")
    sd = load_file(os.path.join(peft_dir, "adapter_model.safetensors"))
    peft_model = student.da3.model.backbone.pretrained
    peft_model.load_state_dict(sd, strict=False)
    pt_path = os.path.join(ckpt_dir, "c2m_final_lora.pt")
    if os.path.exists(pt_path):
        state = torch.load(pt_path, map_location="cpu")
        if "camera_token" in state:
            base = P._unwrap_pretrained(student.da3.model.backbone)
            base.camera_token.data.copy_(state["camera_token"].to(base.camera_token.device))


@torch.no_grad()
def infer_scene(api, image_files, export_dir):
    api.inference(
        image=image_files,
        export_dir=export_dir,
        export_format="mini_npz",
        ref_view_strategy="first",
    )
    # wait for the async npz writer
    import zipfile

    result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    deadline = time.time() + 60
    while True:
        try:
            with zipfile.ZipFile(result_path) as z:
                if z.testzip() is None:
                    return result_path
        except (FileNotFoundError, EOFError, OSError, zipfile.BadZipFile):
            pass
        if time.time() > deadline:
            raise TimeoutError(f"export timeout: {result_path}")
        time.sleep(0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CLASSES))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt_root", default=None, help="per-scene ckpt dirs (TTA arm)")
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    ap.add_argument("--work_root", default="workspace/da3_recon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    scenes = args.scenes or sorted(man["scenes"])
    fg_common.set_dataset(args.dataset)
    dataset = make_dataset(args.dataset)

    if args.ckpt_root:
        print(f"Loading student (LoRA hot-swap per scene): {args.model_name}")
        student = P.create_student(args.model_name, device="cuda")
        student.eval()
        api = student.da3
    else:
        print(f"Loading base model: {args.model_name}")
        student = None
        api = P.create_teacher(args.model_name, device="cuda")

    results = {}
    for scene in scenes:
        t0 = time.time()
        sc = man["scenes"][scene]
        eval_frames = sc["eval32_frames"]
        data = fg_common.get_scene_data(scene)
        image_files = [data.image_files[i] for i in eval_frames]

        if student is not None:
            load_scene_lora_(student, os.path.join(args.ckpt_root, scene))
            student.da3.eval()

        export_dir = os.path.join(args.work_root, args.dataset, args.arm, scene)
        os.makedirs(export_dir, exist_ok=True)
        result_path = infer_scene(api, image_files, export_dir)

        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        payload = {
            "extrinsics": np.asarray(data.extrinsics)[eval_frames],
            "intrinsics": np.asarray(data.intrinsics)[eval_frames],
            "image_files": np.array(image_files, dtype=object),
        }
        aux = data.aux
        if getattr(aux, "get", None) and aux.get("mask_files") is not None:
            payload["mask_files"] = np.array([aux["mask_files"][i] for i in eval_frames], dtype=object)
        np.savez_compressed(meta_path, **payload)

        fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        dataset.fuse3d(scene, result_path, fuse_path, "recon_unposed")
        metrics = dataset.eval3d(scene, fuse_path)
        results[scene] = {k: float(v) for k, v in dict(metrics).items()}
        results[scene]["time_s"] = time.time() - t0
        print(f"[{args.arm}/{scene}] fscore={results[scene].get('fscore', float('nan')):.4f} "
              f"overall={results[scene].get('overall', float('nan')):.4f} "
              f"({results[scene]['time_s']:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "arm": args.arm, "scenes": results}, f, indent=2)
    print(f"Wrote {args.out}")
    print("DONE")


if __name__ == "__main__":
    main()
