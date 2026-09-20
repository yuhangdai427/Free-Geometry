#!/usr/bin/env python3
"""Verify actual train/eval frames and settings against the retained DA3 protocol."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry.common import write_json, digest
from self_geometry.data import dataset
from self_geometry.protocol import paper_protocol
from self_geometry.benchmark import metrics_for
from self_geometry.training import identity
from depth_anything_3.bench.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'matrix_plan.json').read_text())
    profile = paper_protocol(plan['config'], plan['methods'])
    if json.loads((root/'protocol.json').read_text()) != profile:
        raise ValueError('Run has no matching formal protocol record')
    canonical, rows = {}, []
    for job in plan['jobs']:
        row = dict(job, status='pending')
        directory = root/job['method']/job['model']/f'seed_{job["seed"]}'/job['dataset']/job['scene']
        stage = 'baseline' if job['method']=='baseline' else 'adapted'
        try:
            key = (job['dataset'], job['scene'])
            if key not in canonical:
                ds = dataset(job['dataset'], plan['config'])
                data = Evaluator._sample_frames(SimpleNamespace(max_frames=100), ds.get_data(job['scene']), job['scene'])
                canonical[key] = list(data.image_files)
            expected = canonical[key]
            row['expected_frames'] = len(expected)
            if not (directory/'manifest.json').exists():
                rows.append(row); continue
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['image_files'] != expected or manifest['sampling_seed'] != 42 or manifest['max_frames'] != 100:
                raise ValueError('Actual ordered RGB manifest differs from official sampling')
            if manifest['fingerprint'] != digest({k:v for k,v in manifest.items() if k!='fingerprint'}):
                raise ValueError('RGB manifest fingerprint does not match its contents')
            if any(Path(r['path']).stat().st_size != r['bytes'] or
                   Path(r['path']).stat().st_mtime_ns != r['mtime_ns'] for r in manifest['images']):
                raise ValueError('RGB files changed after training preparation')
            with np.load(directory/'gt_meta.npz') as data:
                if list(data['image_files']) != expected:
                    raise ValueError('Evaluation GT order differs from train RGB manifest')
            marker = directory/stage/('protocol.json' if stage=='baseline' else 'complete.json')
            if marker.exists():
                state = json.loads(marker.read_text())
                if state['config']['model'] != job['model'] or state['config']['seed'] != job['seed']:
                    raise ValueError('Prediction model/seed differs from planned job')
                paper_protocol(state['config'], [job['method']])
                if job['method'] == 'test3r' and plan['config'].get('test3r_max_triplets') is not None:
                    from self_geometry.comparisons import triplet_schedule
                    order, settings = triplet_schedule(len(expected), state['config'])
                    sampled = json.loads((directory/stage/'triplets.json').read_text())
                    if (state['config'].get('test3r_max_triplets') != plan['config']['test3r_max_triplets']
                            or sampled['order'] != order or state['settings'] != settings
                            or state['steps'] != settings['microsteps']
                            or state['updates'] != settings['expected_updates']):
                        raise ValueError('Actual Test3R sample/schedule differs from the capped plan')
                if state['identity'] != digest(identity(state['config'], manifest)):
                    raise ValueError('Prediction/checkpoint identity does not match train manifest')
                with np.load(directory/stage/'exports/mini_npz/results.npz') as data:
                    for field in ('depth', 'conf', 'extrinsics', 'intrinsics'):
                        if len(data[field]) != len(expected) or not np.isfinite(data[field]).all():
                            raise ValueError(f'Invalid final full-scene prediction: {field}')
                row['status'] = 'predicted'
                if stage == 'adapted':
                    row['optimizer_updates'] = state['updates']
                    row['selected_updates'] = state.get('selected_updates', state['updates'])
            metrics_path = directory/stage/'metrics.json'
            if metrics_path.exists():
                if row['status'] != 'predicted':
                    raise ValueError('Metrics without a verified completed prediction')
                metrics = json.loads(metrics_path.read_text())
                if not all(k in metrics and np.isfinite(metrics[k]) for k in metrics_for(job['dataset'])):
                    raise ValueError('Missing/nonfinite official pose or reconstruction metrics')
                row['status'] = 'evaluated'
        except Exception as exc:
            row.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    invalid = sum(r['status']=='invalid' for r in rows)
    complete = sum(r['status']=='evaluated' for r in rows)
    write_json(root/'protocol_audit.json', dict(profile=profile['profile'], total=len(rows),
               invalid=invalid, evaluated=complete, complete=complete==len(rows), rows=rows))
    print(f'{complete}/{len(rows)} fully evaluated; {invalid} protocol violations; remaining cells pending')
    for row in rows:
        if row['status']=='invalid': print(row)
    return int(invalid > 0 or (args.require_complete and complete != len(rows)))


if __name__ == '__main__':
    sys.exit(main())
