#!/usr/bin/env python3
"""Finish existing exports without loading training models or changing budgets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry import ROOT
from self_geometry.common import write_json
from self_geometry.ram_limits import enter_runner_scope, scope_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Method root containing MODEL/seed_N')
    p.add_argument('--worker-ram-gib', type=int, default=32)
    a = p.parse_args()
    root = a.root.resolve()
    jobs = []
    for manifest in root.glob('*/seed_*/**/manifest.json'):
        directory = manifest.parent
        m = json.loads(manifest.read_text())
        stage = directory/'adapted'
        if (stage/'metrics.json').exists() or not (stage/'complete.json').exists():
            continue
        folder = next(x for x in directory.parents if x.name.startswith('seed_'))
        cfg = folder.parent/f'config_{folder.name}.json'
        jobs.append((m['dataset'] == 'eth3d', directory, folder, cfg, m))
    state = dict(pid=os.getpid(), status='running', jobs={}, started=time.time())
    status = root/'evaluation_resume.json'
    write_json(status, state)
    for _, directory, folder, cfg, m in sorted(jobs, key=lambda x: (x[0], str(x[1]))):
        key = str(directory.relative_to(root))
        commands = ['evaluate'] if m['dataset'] != 'dtu' else ['fuse', 'score']
        for command in commands:
            state['active'] = key+'/'+command
            write_json(status, state)
            cmd = [sys.executable, str(ROOT/'scripts/run.py'), command, '--config', str(cfg),
                   '--output', str(folder), '--dataset', m['dataset'], '--scene', m['scene'], '--stage', 'adapted']
            log = directory/f'resume_{command}.log'
            tick = time.time()
            print('RUN', key, command, flush=True)
            with log.open('a') as f:
                code = subprocess.run(scope_command(cmd, a.worker_ram_gib), cwd=ROOT,
                                      stdout=f, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, PYTHONUNBUFFERED='1',
                                               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')).returncode
            state['jobs'][state['active']] = dict(exit_code=code, seconds=time.time()-tick, log=str(log))
            write_json(status, state)
            if code:
                break
    state.update(active=None, status='finished_with_failures' if any(j['exit_code'] for j in state['jobs'].values()) else 'complete')
    write_json(status, state)


if __name__ == '__main__':
    enter_runner_scope()
    main()
