#!/usr/bin/env python3
"""Single-model reference runner with separate processes for each seed."""
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
from self_geometry.cache import reuse
from self_geometry.common import config, write_json
from self_geometry.data import dataset


def build_plan(c, names, seeds, first_only=False):
    scenes = {name: list(dataset(name, c).SCENES) for name in names}
    if first_only:
        scenes = {name: values[:1] for name, values in scenes.items()}
    jobs = [dict(seed=seed, dataset=name, scene=scene)
            for seed in seeds for name in names for scene in scenes[name]]
    return dict(config=c, seeds=seeds, datasets=names, scenes=scenes, jobs=jobs,
                adaptations=len(jobs), first_only=first_only, sampling_seed=42)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'artifacts/separate_seeds')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--config', type=Path)
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--first-only', action='store_true', help='Smoke/debug subset; never a full benchmark')
    p.add_argument('--dry-run', action='store_true', help='Write immutable plan and commands without model execution')
    p.add_argument('--reuse-from', type=Path, default=ROOT / 'artifacts/main')
    p.add_argument('--no-reuse', action='store_true')
    p.add_argument('--eval-workers', type=int, default=2)
    p.add_argument('--skip-evaluation', action='store_true')
    a = p.parse_args(argv)
    if len(set(a.seeds)) != len(a.seeds) or any(s < 0 or s >= 2**32 for s in a.seeds):
        p.error('Seeds must be distinct integers in [0, 2**32)')
    if len(set(a.datasets)) != len(a.datasets):
        p.error('Datasets must be distinct')
    if a.eval_workers < 1:
        p.error('--eval-workers must be positive')
    c = config(a.config, a.set)
    # Individual seed values are specified by --seeds, never by a hidden override.
    c['seed'] = a.seeds[0]
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'experiments.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = build_plan(c, a.datasets, a.seeds, a.first_only)
    plan_path = root / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Experiment plan differs; use a new --root')
    write_json(plan_path, plan)
    print(f'{len(a.datasets)} datasets; {sum(map(len, plan["scenes"].values()))} scenes; '
          f'seeds={a.seeds}; {plan["adaptations"]} adaptations', flush=True)
    commands = []
    for seed in a.seeds:
        dest = root / f'seed_{seed}'
        seed_config = dict(c, seed=seed)
        cfg = root / f'config_seed_{seed}.json'  # JSON is a YAML subset.
        write_json(cfg, seed_config)
        cmd = [sys.executable, str(ROOT / 'scripts/campaign.py'), '--output', str(dest),
               '--config', str(cfg), '--datasets', *a.datasets, '--eval-workers', str(a.eval_workers)]
        if a.first_only:
            cmd.append('--first-only')
        if a.skip_evaluation:
            cmd.append('--skip-evaluation')
        commands.append(dict(seed=seed, command=cmd))
    write_json(root / 'commands.json', commands)
    if a.dry_run:
        print(plan_path)
        return 0
    records = []
    status = dict(pid=os.getpid(), started=time.time(), status='running', runs=records)
    for item in commands:
        seed = item['seed']
        status['active_seed'] = seed
        write_json(root / 'experiments.json', status)
        dest = root / f'seed_{seed}'
        seed_config = dict(c, seed=seed)
        if not a.no_reuse:
            sources = [a.reuse_from.resolve(), *(root / f'seed_{s}' for s in a.seeds if s != seed)]
            for source in sources:
                reused = reuse(source, dest, seed_config, completed=True)
                if reused:
                    print(f'seed {seed}: reused {reused} scenes from {source}', flush=True)
        tick = time.time()
        print('SEED', seed, flush=True)
        ret = subprocess.run(item['command'], cwd=ROOT).returncode
        records.append(dict(seed=seed, exit_code=ret, seconds=time.time() - tick))
        # A failed scene/dataset/seed never prevents the next seed from running.
        write_json(root / 'experiments.json', status)
        subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_seeds_separately.py'), '--root', str(root)], check=False)
    failed = any(r['exit_code'] for r in records)
    status.update(status='finished_with_failures' if failed else ('training_only' if a.skip_evaluation else 'complete'),
                  finished=time.time(), active_seed=None)
    write_json(root / 'experiments.json', status)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
