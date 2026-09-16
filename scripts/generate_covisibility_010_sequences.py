#!/usr/bin/env python3
"""Generate shared train/eval 8V sequences from the Co-visibility 0.1 subset."""
import argparse
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "src"), os.path.join(os.path.dirname(__file__), "..", "src", "vggt")]
from vggt.vggt.test_time_adaption.covisibility_sequences import write_sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="artifacts/covisibility_all_datasets_0025/nonoverlap_010_counts.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=["eth3d", "scannetpp"], required=True)
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--samples-per-scene", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    count = write_sequences(args.selection, args.output, args.datasets, args.split, args.samples_per_scene, args.seed, args.epochs)
    print(f"Wrote {count} shared {args.split} sequences to {args.output}")


if __name__ == "__main__":
    main()
