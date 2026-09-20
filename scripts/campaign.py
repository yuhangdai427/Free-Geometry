#!/usr/bin/env python3
"""Restartable single-seed campaign; each scene failure is isolated."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry import ROOT
from self_geometry.benchmark import DATASETS
from self_geometry.common import config, write_json
from self_geometry.data import dataset
from self_geometry.report import report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/main')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--first-only', action='store_true')
    p.add_argument('--config', type=Path)
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--eval-workers', type=int, default=2)
    a = p.parse_args(argv)
    if a.eval_workers < 1:
        p.error('--eval-workers must be positive')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.config:
        a.config = a.config.resolve()
    os.chdir(ROOT)
    lockfile = (a.output / 'campaign.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c = config(a.config, a.set)
    status_path = a.output / 'campaign.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'jobs': {}, 'started': time.time()}
    if 'config' in status and status['config'] != c:
        raise ValueError('Campaign configuration mismatch; choose a new output')
    status.update(config=c, active=[], pid=os.getpid(), datasets=a.datasets, first_only=a.first_only)
    status.pop('finished', None)
    write_json(status_path, status)
    mutex = threading.Lock()
    base = [sys.executable, str(ROOT / 'scripts/run.py')]
    options = ['--output', str(a.output)]
    if a.config:
        options += ['--config', str(a.config)]
    for override in a.set:
        options += ['--set', override]
    env = dict(os.environ, OMP_NUM_THREADS=str(c['threads']), OPENBLAS_NUM_THREADS=str(c['threads']),
               MKL_NUM_THREADS=str(c['threads']), PYTHONUNBUFFERED='1', TORCH_HOME=str(ROOT / 'weights'),
               HF_HUB_OFFLINE='1', LOKY_MAX_CPU_COUNT=str(c['threads']),
               PYTORCH_ALLOC_CONF='expandable_segments:True')

    def run(name, scene, command, extra):
        key = '/'.join([name, scene, command] + extra)
        artifact = a.output / name / scene / {
            'baseline': 'baseline/protocol.json', 'adapt': 'adapted/complete.json',
            'evaluate': (extra[-1] + '/metrics.json') if extra else '',
        }[command]
        with mutex:
            ready = artifact.exists()
            if command in ('baseline', 'adapt'):
                stage = 'baseline' if command == 'baseline' else 'adapted'
                ready = ready and (a.output / name / scene / stage / 'exports/mini_npz/results.npz').exists()
            if status['jobs'].get(key, {}).get('exit_code') == 0 and ready:
                return 0
            status['active'].append(key)
            write_json(status_path, status)
        log = a.output / name / scene / (command + ('_' + extra[-1] if command == 'evaluate' else '') + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = base + [command] + options + ['--dataset', name, '--scene', scene] + extra
        print('RUN', key, flush=True)
        start = time.time()
        try:
            with log.open('a') as f:
                ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        except OSError as exc:
            with log.open('a') as f:
                f.write(f'Could not launch stage: {exc}\n')
            ret = 127
        with mutex:
            status['active'].remove(key)
            status['jobs'][key] = dict(exit_code=ret, seconds=time.time() - start, log=str(log))
            write_json(status_path, status)
        if ret:
            print('FAILED (continuing)', key, ret, flush=True)
        return ret

    with ThreadPoolExecutor(max_workers=a.eval_workers) as pool:
        pending = []
        for name in a.datasets:
            # Official DTU fusion uses CUDA. Drain CPU evaluators, then execute
            # DTU inference/adaptation/fusion serially to bound our GPU usage.
            if name == 'dtu':
                for future in pending:
                    future.result()
                pending.clear()
            scenes = dataset(name, c).SCENES
            if a.first_only:
                scenes = scenes[:1]
            for scene in scenes:
                if run(name, scene, 'baseline', []):
                    continue
                if not a.skip_evaluation:
                    if name == 'dtu':
                        run(name, scene, 'evaluate', ['--stage', 'baseline'])
                    else:
                        pending.append(pool.submit(run, name, scene, 'evaluate', ['--stage', 'baseline']))
                if run(name, scene, 'adapt', ['--resume']):
                    continue
                if not a.skip_evaluation:
                    if name == 'dtu':
                        run(name, scene, 'evaluate', ['--stage', 'adapted'])
                    else:
                        pending.append(pool.submit(run, name, scene, 'evaluate', ['--stage', 'adapted']))
            # Limit evaluator backlog to one dataset and make intermediate reports.
            for future in pending:
                future.result()
            pending.clear()
            report(a.output, c)
    status.update(finished=time.time(), current=None)
    write_json(status_path, status)
    return int(any(j['exit_code'] for j in status['jobs'].values()))


if __name__ == '__main__':
    sys.exit(main())
