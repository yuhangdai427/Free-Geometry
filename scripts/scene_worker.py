#!/usr/bin/env python3
"""One GPU process per model/scene; share frozen model and inputs across seeds."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry import ROOT
from self_geometry.common import config, digest, write_json
from self_geometry.data import prepare
from self_geometry.cache import copy_cached
from self_geometry.model import load_model, load_images, remove_lora
from self_geometry.training import baseline, adapt, identity


def reuse_baseline(source, target, c):
    if source == target or not (source / 'baseline/protocol.json').exists():
        return False
    manifest = json.loads((source / 'manifest.json').read_text())
    protocol = json.loads((source / 'baseline/protocol.json').read_text())
    old = protocol['config']
    from self_geometry.cache import BASELINE_KEYS
    if old.get('model', 'vggt') != c['model'] or any(old[k] != c[k] for k in BASELINE_KEYS):
        return False
    if old.get('ref_view_strategy', 'first') != c['ref_view_strategy']:
        return False
    if protocol['identity'] != digest(identity(old, manifest)):
        return False
    if not (source / 'baseline/exports/mini_npz/results.npz').exists():
        return False
    if (target / 'baseline/protocol.json').exists():
        return False
    if any(not Path(r['path']).exists() or Path(r['path']).stat().st_size != r['bytes'] or
           Path(r['path']).stat().st_mtime_ns != r['mtime_ns'] for r in manifest['images']):
        return False
    for filename in ('manifest.json', 'gt_meta.npz', 'baseline/exports/mini_npz/results.npz', 'baseline/metrics.json'):
        if (source / filename).exists():
            copy_cached(source / filename, target / filename)
    if old['keypoints'] == c['keypoints'] and (source / 'matches.pt').exists():
        copy_cached(source / 'matches.pt', target / 'matches.pt')
    protocol.update(identity=digest(identity(c, manifest)), config=c, reused_from=str(source))
    write_json(target / 'baseline/protocol.json', protocol)
    return True


def main(argv=None):
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--reuse-from', type=Path)
    a = p.parse_args(argv)
    c = config(a.config)
    root = a.root.resolve()
    os.environ['TORCH_HOME'] = str(ROOT / 'weights')
    os.environ['HF_HUB_OFFLINE'] = '1'
    records, cache, model = [], {}, None
    model_seconds = 0.
    source = None
    for seed in a.seeds:
        current = dict(c, seed=seed)
        directory = root / f'seed_{seed}' / a.dataset / a.scene
        tick = time.monotonic()
        record = dict(model=c['model'], dataset=a.dataset, scene=a.scene, seed=seed, baseline=False, adapted=False)
        try:
            prepare(a.dataset, a.scene, current, directory)
            if source is None and a.reuse_from:
                reuse_baseline(a.reuse_from / a.dataset / a.scene, directory, current)
            if source is not None:
                reuse_baseline(source, directory, current)
            baseline_ready = (directory / 'baseline/protocol.json').exists() and (directory / 'baseline/exports/mini_npz/results.npz').exists()
            adapted_ready = (directory / 'adapted/complete.json').exists() and (directory / 'adapted/exports/mini_npz/results.npz').exists()
            # Lazy loading avoids loading 5 GB just to skip already complete work.
            if not (baseline_ready and adapted_ready) and model is None:
                start = time.monotonic()
                model = load_model(current)
                model_seconds = time.monotonic() - start
            if model is not None:
                remove_lora(model)
            if not (baseline_ready and adapted_ready) and 'images' not in cache:
                manifest = json.loads((directory / 'manifest.json').read_text())
                cache['images'] = load_images(manifest['image_files'], c['image_size'], c['model']).cuda()
            baseline(current, directory, model=model, images=cache.get('images'))
            record['baseline'] = True
            source = directory
            adapt(current, directory, resume=True, model=model, scene_cache=cache)
            record['adapted'] = True
            record['exit_code'] = 0
        except Exception:
            record.update(exit_code=1, error=traceback.format_exc())
            print(record['error'], flush=True)
            # Release graph/optimizer allocations before continuing to another seed.
            gc.collect()
            torch.cuda.empty_cache()
        record['seconds'] = time.monotonic() - tick
        write_json(directory / 'worker.json', record)
        records.append(record)
        print('SEED_RESULT', json.dumps(record), flush=True)
    write_json(root / 'workers' / a.dataset / a.scene / 'result.json',
               dict(model_load_seconds=model_seconds, runs=records, shared_model=True,
                    shared_images=True, shared_matches=True, shared_initialization=True))
    return int(any(r['exit_code'] for r in records))


if __name__ == '__main__':
    sys.exit(main())
