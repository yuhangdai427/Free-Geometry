#!/usr/bin/env python3
"""DA3/VGGT x baseline/Test3R/Self-Geometry/TCO x 5 datasets x 3 seeds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry import ROOT
from self_geometry.benchmark import DATASETS, SEEDS
from self_geometry.comparisons import METHODS
from self_geometry.common import config, write_json
from self_geometry.data import dataset


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/comparison')
    p.add_argument('--config', type=Path, default=ROOT/'configs/comparison.yaml')
    p.add_argument('--models', nargs='+', choices=['da3', 'vggt'], default=['da3', 'vggt'])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=['baseline', 'self_geometry', 'tco'])
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--first-only', action='store_true')
    p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--protocol', choices=['custom', 'paper'], default='custom')
    p.add_argument('--eval-workers', type=int, default=1)
    p.add_argument('--worker-ram-gib', type=int, default=32)
    p.add_argument('--gpu-workers', type=int, default=2, help='Concurrent scene processes, subject to GPU memory admission')
    p.add_argument('--reuse-model-root', type=Path, help='Reuse compatible frozen baselines from another matrix')
    p.add_argument('--da3-weights', type=Path)
    p.add_argument('--vggt-weights', type=Path)
    a = p.parse_args(argv)
    if not 0 < a.worker_ram_gib <= 72:
        p.error('worker-ram-gib must be in 1..72')
    for key in ('models','methods','datasets','seeds'):
        if len(set(getattr(a,key))) != len(getattr(a,key)):
            p.error('Duplicate ' + key)
    if any(s < 0 or s >= 2**32 for s in a.seeds) or min(a.eval_workers, a.gpu_workers) < 1:
        p.error('Invalid seed or worker count')
    if any(s.split('=')[0] in ('method','model','seed') for s in a.set):
        p.error('Use --methods/--models/--seeds')
    if any(s.split('=')[0] == 'weights' for s in a.set):
        p.error('Use --da3-weights / --vggt-weights')
    c = config(a.config, a.set)
    if a.protocol == 'paper':
        from self_geometry.protocol import paper_protocol
        try:
            protocol = paper_protocol(c, a.methods, a.skip_evaluation)
        except ValueError as exc:
            p.error(str(exc))
    else:
        protocol = dict(profile='custom', note='Not certified as the formal paper protocol')
    if c.get('test3r_vggt_points', 'depth') not in ('native', 'depth'):
        p.error('test3r_vggt_points must be native or depth')
    for key in ('test3r_epochs','test3r_accum','test3r_prompt_size','test3r_pair_batch'):
        if not isinstance(c[key], int) or c[key] < 1:
            p.error(key + ' must be a positive integer')
    for key in ('tco_steps','test3r_max_updates','test3r_max_triplets'):
        if c.get(key) is not None and (type(c[key]) is not int or c[key] < 1):
            p.error(key + ' must be null or a positive integer')
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'matrix.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scenes = {n:list(dataset(n,c).SCENES) for n in a.datasets}
    if a.first_only:
        scenes = {k:v[:1] for k,v in scenes.items()}
    jobs = [dict(model=m, method=t, dataset=d, scene=s, seed=r)
            for m in a.models for t in a.methods for d,ss in scenes.items() for s in ss for r in a.seeds]
    plan = dict(schema=1, models=a.models, methods=a.methods, datasets=a.datasets, seeds=a.seeds,
                scenes=scenes, config=c, first_only=a.first_only, jobs=jobs, cells=len(jobs),
                weights={m:str(getattr(a,m+'_weights').resolve()) if getattr(a,m+'_weights') else None for m in a.models})
    path = root/'matrix_plan.json'
    if a.reuse_model_root:
        plan['reuse_model_root'] = str(a.reuse_model_root.resolve())
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Different experiment plan: use a new --root')
    write_json(path, plan)
    protocol_path = root/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Different protocol profile: use a new --root')
    write_json(protocol_path, protocol)
    print(f'{len(jobs)} model/method/scene/seed cells; baseline predictions are shared across methods/seeds', flush=True)
    state = dict(pid=os.getpid(), started=time.time(), status='running', methods={})
    failed = False
    # Exhaustive Test3R can be much longer, so finish the other methods first.
    for method in ('baseline','self_geometry','tco','test3r'):
        if method not in a.methods:
            continue
        cfg = root / f'config_{method}.json'
        write_json(cfg, dict(c, method=method))
        cmd = [sys.executable, str(ROOT/'scripts/run_full.py'), '--root', str(root/method),
               '--config', str(cfg), '--models', *a.models, '--datasets', *a.datasets,
               '--seeds', *map(str,a.seeds), '--eval-workers', str(a.eval_workers),
               '--gpu-workers', str(a.gpu_workers), '--worker-ram-gib', str(a.worker_ram_gib)]
        if method != 'baseline' and 'baseline' in a.methods:
            cmd += ['--reuse-model-root', str(root/'baseline')]
        elif a.reuse_model_root:
            cmd += ['--reuse-model-root', str(a.reuse_model_root.resolve())]
        for option in ('first_only','skip_evaluation','dry_run'):
            if getattr(a, option):
                cmd.append('--'+option.replace('_','-'))
        for m in a.models:
            if getattr(a,m+'_weights'):
                cmd += ['--'+m+'-weights', str(getattr(a,m+'_weights').resolve())]
        write_json(root/f'command_{method}.json', cmd)
        if not a.dry_run:
            state['active_method'] = method
            write_json(root/'matrix_state.json',state)
        code = subprocess.run(cmd,cwd=ROOT).returncode
        state['methods'][method] = dict(exit_code=code)
        failed |= code != 0
    if not a.dry_run:
        if a.protocol == 'paper':
            audit_code = subprocess.run([
                sys.executable, str(ROOT/'scripts/audit_paper_run.py'),
                '--root', str(root), '--require-complete']).returncode
            state['protocol_audit_exit_code'] = audit_code
            failed |= audit_code != 0
        state.update(status='finished_with_failures' if failed else ('training_only' if a.skip_evaluation else 'complete'),
                     finished=time.time(),active_method=None)
        write_json(root/'matrix_state.json',state)
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_comparison.py'),'--root',str(root)],check=True)
    return int(failed)

if __name__ == '__main__':
    from self_geometry.ram_limits import enter_runner_scope
    enter_runner_scope()
    sys.exit(main())
