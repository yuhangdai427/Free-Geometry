#!/usr/bin/env python3
"""Prepare the Free-Geo checkpoint release folder layout.

Expected output layout:

Free-Geo_ckpts/
├── DA3+Free-Geo/
│   ├── 7scenes/
│   ├── eth3d/
│   ├── hiroom/
│   └── scannetpp/
└── VGGT+Free-Geo/
    ├── 7scenes/
    ├── eth3d/
    ├── hiroom/
    └── scannetpp/

Each dataset directory is populated from a `latest_lora_peft` source folder.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path("/root/autodl-tmp/da3")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Free-Geo_ckpts"

SOURCE_LAYOUT = {
    "DA3+Free-Geo": {
        "hiroom": REPO_ROOT / "checkpoints" / "all_da3" / "hiroom" / "latest_lora_peft",
        "eth3d": REPO_ROOT / "checkpoints" / "all_da3_v2" / "eth3d" / "latest_lora_peft",
        "7scenes": REPO_ROOT / "checkpoints" / "all_da3_v2" / "7scenes" / "latest_lora_peft",
        "scannetpp": REPO_ROOT / "checkpoints" / "all_da3_v2" / "scannetpp" / "latest_lora_peft",
    },
    "VGGT+Free-Geo": {
        "hiroom": REPO_ROOT / "checkpoints" / "all_vggt" / "hiroom" / "latest_lora_peft",
        "eth3d": REPO_ROOT / "checkpoints" / "all_vggt_v3" / "eth3d" / "latest_lora_peft",
        "7scenes": REPO_ROOT / "checkpoints" / "all_vggt_v3" / "7scenes" / "latest_lora_peft",
        "scannetpp": REPO_ROOT / "checkpoints" / "all_vggt_v3" / "scannetpp" / "latest_lora_peft",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Free-Geo checkpoint folder structure from local PEFT adapters."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copy operations without creating or modifying files.",
    )
    return parser.parse_args()


def validate_sources() -> None:
    missing = []
    for model_name, datasets in SOURCE_LAYOUT.items():
        for dataset_name, source_dir in datasets.items():
            if not source_dir.is_dir():
                missing.append(f"{model_name}/{dataset_name}: {source_dir}")

    if missing:
        missing_text = "\n".join(f"  - {entry}" for entry in missing)
        raise FileNotFoundError(f"Missing expected source adapter directories:\n{missing_text}")


def copy_adapter_dir(source_dir: Path, target_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] {source_dir} -> {target_dir}")
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)
    print(f"Copied {source_dir} -> {target_dir}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()

    validate_sources()

    for model_name, datasets in SOURCE_LAYOUT.items():
        for dataset_name, source_dir in datasets.items():
            target_dir = output_root / model_name / dataset_name
            copy_adapter_dir(source_dir, target_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Finished creating checkpoint layout under: {output_root}")


if __name__ == "__main__":
    main()
