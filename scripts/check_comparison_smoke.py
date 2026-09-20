#!/usr/bin/env python3
"""Audit every matrix cell and compute official pose metrics, without 3D fusion.

This is a smoke check, not a replacement for the full benchmark evaluator.
Ground truth is read only here, after training, and never used for adaptation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry.common import write_json
from depth_anything_3.bench.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = json.loads((args.root / 'matrix_plan.json').read_text())
    rows = []
    for job in plan['jobs']:
        directory = (args.root / job['method'] / job['model'] /
                     f'seed_{job["seed"]}' / job['dataset'] / job['scene'])
        stage = 'baseline' if job['method'] == 'baseline' else 'adapted'
        row = dict(job)
        try:
            marker = 'protocol.json' if stage == 'baseline' else 'complete.json'
            if not (directory / stage / marker).is_file():
                raise ValueError('Missing completion marker')
            manifest = json.loads((directory / 'manifest.json').read_text())
            prediction = directory / stage / 'exports/mini_npz/results.npz'
            with np.load(prediction) as data:
                for key in ('depth', 'conf', 'extrinsics', 'intrinsics'):
                    value = data[key]
                    if len(value) != len(manifest['image_files']):
                        raise ValueError(f'{key}: frame count mismatch')
                    if not np.isfinite(value).all():
                        raise ValueError(f'{key}: nonfinite values')
                if not (data['depth'] > 0).all():
                    raise ValueError('Nonpositive depth')
                row['depth_range'] = [float(data['depth'].min()), float(data['depth'].max())]
            with np.load(directory / 'gt_meta.npz') as data:
                gt = dict(data)
            if list(gt['image_files']) != manifest['image_files']:
                raise ValueError('GT image order differs from manifest')
            metrics = Evaluator._compute_pose_with_gt(None, str(prediction), gt)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError('Nonfinite pose metrics')
            row.update(passed=True, pose_metrics={k: float(v) for k, v in metrics.items()})
        except Exception as exc:
            row.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    passed = sum(row['passed'] for row in rows)
    report = dict(scope='export integrity and official pose only; no 3D fusion/evaluation',
                  passed=passed, total=len(rows), all_passed=passed == len(rows), rows=rows)
    write_json(args.root / 'smoke_checks.json', report)
    print(f'{passed}/{len(rows)} export and pose checks passed')
    for row in rows:
        if not row['passed']:
            print(row)
    return int(passed != len(rows))


if __name__ == '__main__':
    sys.exit(main())
