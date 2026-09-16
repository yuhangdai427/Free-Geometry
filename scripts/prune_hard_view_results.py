#!/usr/bin/env python3
"""Remove bulky per-sequence predictions only after a five-arm report exists."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_REQUIRED_ARMS = {"base_8v", "base_4v", "base_8v_extract_4v", "lora_4v", "lora_8v"}


def prune(result_root: Path, dataset: str, required_arms: set[str]) -> bool:
    metrics_path = result_root / "pose_metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
        arms = set(metrics[dataset])
    except (json.JSONDecodeError, KeyError):
        return False
    if not required_arms.issubset(arms):
        return False
    payload = result_root / dataset
    if payload.is_dir():
        shutil.rmtree(payload)
    print(f"[pruned] {payload}; retained {metrics_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=["eth3d", "scannetpp"])
    parser.add_argument("--required-arms", nargs="+", default=sorted(DEFAULT_REQUIRED_ARMS))
    args = parser.parse_args()
    raise SystemExit(0 if prune(args.result_root, args.dataset, set(args.required_arms)) else 1)


if __name__ == "__main__":
    main()
