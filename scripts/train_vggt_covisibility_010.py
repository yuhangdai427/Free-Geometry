#!/usr/bin/env python3
"""Train one VGGT LoRA on shared Co-visibility-0.1 8V/4V sequences.

This is intentionally a thin, isolated entry point: it reuses the established
Free-Geometry objective but replaces only its input dataset.  No benchmark
dataset loader is changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "vggt")]


class CovisibilitySequenceDataset(Dataset):
    """Uses epoch-indexed records and preserves the required 0,2,4,6 views."""

    manifest_path = ""

    def __init__(self, dataset_name, image_size=504, student_indices=None, **_unused):
        with open(self.manifest_path, encoding="utf-8") as handle:
            all_records = [json.loads(line) for line in handle]
        self.records = [record for record in all_records if record["dataset"] == dataset_name]
        if not self.records:
            raise ValueError(f"No {dataset_name} records in {self.manifest_path}")
        self.image_size = image_size
        self.student_indices = student_indices or [0, 2, 4, 6]
        if self.student_indices != [0, 2, 4, 6]:
            raise ValueError("This protocol fixes --student_indices to 0 2 4 6")
        self.current_epoch = 0
        self.epochs = sorted({record.get("epoch", 0) for record in self.records})
        print(f"Loaded {len(self.records)} Co-visibility records ({len(self.epochs)} manifest epochs)")

    def _epoch_records(self):
        epoch = self.epochs[self.current_epoch % len(self.epochs)]
        return [record for record in self.records if record.get("epoch", 0) == epoch]

    def __len__(self):
        return len(self._epoch_records())

    def _image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        scale = self.image_size / max(h, w)
        new_h = max(14, round((h * scale) / 14) * 14)
        new_w = max(14, round((w * scale) / 14) * 14)
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

    def __getitem__(self, index):
        record = self._epoch_records()[index]
        if len(record["eight_image_files"]) != 8 or len(set(record["eight_frame_ids"])) != 8:
            raise ValueError(
                f"Strict 8V training requires eight unique frames: "
                f"{record['dataset']}/{record['scene']}"
            )
        teacher = torch.stack([self._image(path) for path in record["eight_image_files"]])
        # Validate the serialized contract before exposing data to the trainer.
        expected = [record["eight_image_files"][i] for i in [0, 2, 4, 6]]
        if record["four_image_files"] != expected:
            raise ValueError(f"Broken 4V contract: {record['dataset']}/{record['scene']}")
        return {"teacher_images": teacher, "student_images": teacher[[0, 2, 4, 6]], "scene": record["scene"]}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sequence-manifest", required=True)
    known, remaining = parser.parse_known_args()
    CovisibilitySequenceDataset.manifest_path = known.sequence_manifest

    # Reuse the existing, versioned loss/training loop with the isolated input.
    import train_vggt as trainer
    trainer.VGGTFreeGeometryDataset = CovisibilitySequenceDataset
    sys.argv = [sys.argv[0], *remaining]
    trainer.main()


if __name__ == "__main__":
    main()
