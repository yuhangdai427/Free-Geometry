#!/usr/bin/env python3
"""Evaluate each scene's single trained checkpoint on multiple RGB samples."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry import ROOT
from self_geometry.common import digest, write_json, seed_all
from self_geometry.data import prepare
from self_geometry.ram_limits import enter_runner_scope, scope_command


def frame_key(manifest):
    # The seed label is not part of inference identity; ordered RGB files are.
    return digest(manifest['images'])


def infer_samples(source, output, seeds):
    import torch
    from self_geometry.training import identity, prediction_path, export
    from self_geometry.model import load_model, load_images, predict, inject_lora, load_trainable
    from self_geometry.comparisons import install_tco, install_test3r
    manifest = json.loads((source/'manifest.json').read_text())
    stage = 'adapted' if (source/'adapted/complete.json').exists() else 'baseline'
    marker = source/stage/('complete.json' if stage == 'adapted' else 'protocol.json')
    completed = json.loads(marker.read_text())
    c = completed['config']
    if c.get('method', 'self_geometry') != 'baseline' and stage != 'adapted':
        raise ValueError('Adaptation is incomplete; never substitute baseline for it')
    if completed['identity'] != digest(identity(c, manifest)):
        raise ValueError('Source identity changed')
    if any(Path(item['path']).stat().st_size != item['bytes'] or
           Path(item['path']).stat().st_mtime_ns != item['mtime_ns'] for item in manifest['images']):
        raise ValueError('Source RGB files changed since adaptation')
    checkpoint = None
    if stage == 'adapted':
        checkpoint = source/stage/('best.pt' if c['method'] == 'self_geometry' else 'final.pt')
        with checkpoint.open('rb') as f:
            checkpoint_id = hashlib.file_digest(f, 'sha256').hexdigest()
    else:
        checkpoint_id = completed['identity']
    known = {}
    if prediction_path(source, stage).exists():
        known[frame_key(manifest)] = source
    model, records = None, []
    seed_all(c['seed'], c['threads'])
    for seed in seeds:
        directory = output/f'eval_seed_{seed}'/manifest['dataset']/manifest['scene']
        selected = prepare(manifest['dataset'], manifest['scene'], c, directory, sampling_seed=seed)
        key = frame_key(selected)
        stamp = dict(checkpoint_sha256=checkpoint_id, source=str(source), stage=stage,
                     training_seed=c['seed'], sampling_seed=seed, frame_identity=key,
                     config=c, manifest=selected['fingerprint'])
        canonical = known.get(key, directory)
        stamp['canonical_directory'] = str(canonical)
        path = directory/'evaluation_ref.json'
        if path.exists() and json.loads(path.read_text()) != stamp:
            raise ValueError(f'Evaluation checkpoint/manifest changed: {path}')
        if canonical == directory and not (path.exists() and prediction_path(directory, stage).exists()):
            if model is None:
                model = load_model(c)
                if checkpoint is not None:
                    if c['method'] == 'self_geometry':
                        inject_lora(model, c)
                    elif c['method'] == 'tco':
                        install_tco(model, c)
                    else:
                        install_test3r(model, c)
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    load_trainable(model, state['parameters'] if c['method'] == 'self_geometry' else state)
                model.eval()
            images = load_images(selected['image_files'], c['image_size'], c['model']).cuda()
            with torch.no_grad():
                result = predict(model, images, c)
            export(result, prediction_path(directory, stage))
            del images, result
        known[key] = canonical
        write_json(path, stamp)
        records.append(dict(seed=seed, directory=str(directory), canonical=str(canonical), stage=stage))
    write_json(output/'workers'/manifest['dataset']/manifest['scene']/'evaluation_samples.json', records)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Single-training comparison root')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--eval-seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--worker-ram-gib', type=int, default=32)
    p.add_argument('--scene-worker', action='store_true')
    a = p.parse_args()
    if len(set(a.eval_seeds)) != len(a.eval_seeds) or any(s < 0 or s >= 2**32 for s in a.eval_seeds):
        p.error('Evaluation seeds must be distinct uint32 values')
    source, output = a.source.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if a.scene_worker:
        from self_geometry.gpu_queue import reserve_gpu
        with reserve_gpu():
            infer_samples(source, output, a.eval_seeds)
        return 0
    lock = (output/'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((source/'matrix_plan.json').read_text())
    if len(plan['seeds']) != 1:
        raise ValueError('Expected one training seed, not independent training replicas')
    specification = dict(source=str(source), training_seed=plan['seeds'][0], eval_seeds=a.eval_seeds,
                         deduplication='ordered RGB file identities + fixed checkpoint')
    path = output/'plan.json'
    if path.exists() and json.loads(path.read_text()) != specification:
        raise ValueError('Evaluation plan changed; use a new output')
    write_json(path, specification)
    state = dict(pid=os.getpid(), status='running', jobs={}, started=time.time())
    state_path = output/'state.json'
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    def run(key, command):
        log = output/'logs'/f'{key}.log'; log.parent.mkdir(parents=True, exist_ok=True)
        state['active'] = key; write_json(state_path, state)
        tick = time.time()
        with log.open('a') as f:
            code = subprocess.run(scope_command(command, a.worker_ram_gib), cwd=ROOT,
                                  env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        state['jobs'][key] = dict(exit_code=code, seconds=time.time()-tick, log=str(log))
        write_json(state_path, state)
        return code
    for job in plan['jobs']:
        method, model, name, scene = [job[k] for k in ('method', 'model', 'dataset', 'scene')]
        source_scene = source/method/model/f'seed_{job["seed"]}'/name/scene
        target = output/method/model
        key = '/'.join([method, model, name, scene])
        cmd = [sys.executable, str(Path(__file__).resolve()), '--scene-worker', '--source', str(source_scene),
               '--output', str(target), '--eval-seeds', *map(str, a.eval_seeds)]
        if run(key+'/inference', cmd):
            continue
        records = json.loads((target/'workers'/name/scene/'evaluation_samples.json').read_text())
        visited = set()
        for row in records:
            canonical = Path(row['canonical']); stage = row['stage']
            if canonical in visited:
                continue
            visited.add(canonical)
            cfg = source/method/model/f'config_seed_{job["seed"]}.json'
            # All directories end in DATASET/SCENE, including nested HiRoom names.
            folder = canonical
            for _ in Path(name, scene).parts:
                folder = folder.parent
            from run_full import metrics_complete
            if metrics_complete(canonical/stage, name):
                continue
            for command in (['fuse', 'score'] if name == 'dtu' else ['evaluate']):
                cmd = [sys.executable, str(ROOT/'scripts/run.py'), command, '--config', str(cfg),
                       '--output', str(folder), '--dataset', name, '--scene', scene, '--stage', stage]
                if run(key+f'/eval_{row["seed"]}_{command}', cmd):
                    break
    state.update(active=None, status='finished_with_failures' if any(j['exit_code'] for j in state['jobs'].values()) else 'complete')
    write_json(state_path, state)
    from summarize_eval_seeds import summarize
    summarize(output)
    return int(state['status'] != 'complete')


if __name__ == '__main__':
    enter_runner_scope()
    sys.exit(main())
