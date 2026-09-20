#!/usr/bin/env python3
"""DA3-Giant + VGGT × five datasets × three seeds, with shared scene workers."""
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
from self_geometry.benchmark import DATASETS, SEEDS
from self_geometry.common import config, write_json
from self_geometry.data import dataset
from self_geometry.cache import copy_cached

MODELS = ('da3', 'vggt')


def build_plan(configs, names, seeds, first_only=False):
    scenes = {name: list(dataset(name, next(iter(configs.values()))).SCENES) for name in names}
    if first_only:
        scenes = {name: values[:1] for name, values in scenes.items()}
    jobs = [dict(model=model, seed=seed, dataset=name, scene=scene)
            for model in configs for name in names for scene in scenes[name] for seed in seeds]
    return dict(configs=configs, models=list(configs), seeds=seeds, datasets=names, scenes=scenes,
                jobs=jobs, adaptations=sum(configs[j['model']].get('method') != 'baseline' for j in jobs), first_only=first_only, sampling_seed=42,
                scene_workers=len(jobs) // len(seeds), schema=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'artifacts/full_dual')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--config', type=Path)
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--vggt-weights', type=Path)
    p.add_argument('--da3-weights', type=Path)
    p.add_argument('--first-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--eval-workers', type=int, default=2)
    p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--no-reuse', action='store_true')
    p.add_argument('--reuse-model-root', type=Path, help='Import frozen baselines from ROOT/MODEL/seed_FIRST')
    p.add_argument('--reuse-from', type=Path, default=ROOT / 'artifacts/main')
    a = p.parse_args(argv)
    for label, values in [('models', a.models), ('datasets', a.datasets), ('seeds', a.seeds)]:
        if len(set(values)) != len(values):
            p.error(f'Duplicate {label}')
    if any(s < 0 or s >= 2**32 for s in a.seeds) or a.eval_workers < 1:
        p.error('Invalid seed or eval worker count')
    if any(x.split('=')[0] in ('model', 'seed') for x in a.set):
        p.error('Use --models and --seeds')
    if len(a.models) > 1 and any(x.split('=')[0] == 'weights' for x in a.set):
        p.error('Use --da3-weights / --vggt-weights when running both models')
    configs = {}
    for model in a.models:
        c = config(a.config, a.set, model=model)
        c['seed'] = a.seeds[0]
        weights = getattr(a, model + '_weights')
        if weights:
            c['weights'] = str(weights.resolve())
        configs[model] = c
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = build_plan(configs, a.datasets, a.seeds, a.first_only)
    path = root / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Different experiment plan; use a new --root')
    write_json(path, plan)
    print(f'{len(configs)} models × {len(a.datasets)} datasets × {len(a.seeds)} seeds: '
          f'{plan["adaptations"]} adaptations in {plan["scene_workers"]} shared scene workers', flush=True)
    commands = []
    for model, c in configs.items():
        cfg = root / f'config_{model}.json'
        write_json(cfg, c)
        for name, scenes in plan['scenes'].items():
            for scene in scenes:
                cmd = [sys.executable, str(ROOT / 'scripts/scene_worker.py'), '--config', str(cfg),
                       '--root', str(root / model), '--dataset', name, '--scene', scene,
                       '--seeds', *map(str, a.seeds)]
                if not a.no_reuse and a.reuse_model_root:
                    cmd += ['--reuse-from', str(a.reuse_model_root.resolve() / model / f'seed_{a.seeds[0]}')]
                elif not a.no_reuse and model == 'vggt':
                    cmd += ['--reuse-from', str(a.reuse_from.resolve())]
                commands.append(dict(model=model, dataset=name, scene=scene, command=cmd))
    write_json(root / 'commands.json', commands)
    if a.dry_run:
        print(path)
        return 0
    state_path = root / 'suite.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': {}}
    state.update(pid=os.getpid(), started=time.time(), status='running', active=[])
    mutex = threading.Lock()
    failures = []

    def run(key, cmd, log, env):
        with mutex:
            state['active'].append(key)
            write_json(state_path, state)
        log.parent.mkdir(parents=True, exist_ok=True)
        tick = time.time()
        print('RUN', key, flush=True)
        try:
            with log.open('a') as f:
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        except OSError as exc:
            log.write_text(str(exc))
            code = 127
        with mutex:
            state['active'].remove(key)
            state['jobs'][key] = dict(exit_code=code, seconds=time.time() - tick, log=str(log))
            if code:
                failures.append(key)
            write_json(state_path, state)
        return code

    def evaluate(item, seed, stage, env):
        model, name, scene = (item[k] for k in ('model', 'dataset', 'scene'))
        folder = root / model / f'seed_{seed}'
        directory = folder / name / scene
        if not (directory / stage / 'exports/mini_npz/results.npz').exists():
            return
        c = dict(configs[model], seed=seed)
        cfg = root / model / f'config_seed_{seed}.json'
        command = 'score' if name == 'dtu' else 'evaluate'
        cmd = [sys.executable, str(ROOT / 'scripts/run.py'), command, '--config', str(cfg),
               '--output', str(folder), '--dataset', name, '--scene', scene, '--stage', stage]
        key = f'{model}/{name}/{scene}/seed_{seed}/{stage}_evaluate'
        # DTU point clouds were fused serially on GPU; distance scoring is CPU.
        score_env = dict(env, CUDA_VISIBLE_DEVICES='') if name == 'dtu' else env
        code = run(key, cmd, directory / f'evaluate_{stage}.log', score_env)
        # Frozen baseline and evaluation RNG are identical across seeds.
        if code == 0 and stage == 'baseline':
            for other in a.seeds:
                target = root / model / f'seed_{other}' / name / scene
                if (target / 'baseline/protocol.json').exists() and other != seed:
                    copy_cached(directory / 'baseline/metrics.json', target / 'baseline/metrics.json')

    for model, c in configs.items():
        for seed in a.seeds:
            write_json(root / model / f'config_seed_{seed}.json', dict(c, seed=seed))
    with ThreadPoolExecutor(max_workers=a.eval_workers) as pool:
        pending = []
        for item in commands:
            model, name, scene = (item[k] for k in ('model', 'dataset', 'scene'))
            # Bound the CPU queue without idling the GPU at dataset boundaries.
            # DTU fusion still runs serially on this thread; only scoring joins
            # the CPU pool. Evaluation math and scene sampling are unchanged.
            while len(pending) >= 2 * a.eval_workers:
                pending.pop(0).result()
            c = configs[model]
            env = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
                       TORCH_HOME=str(ROOT / 'weights'), OMP_NUM_THREADS=str(c['threads']),
                       OPENBLAS_NUM_THREADS=str(c['threads']), MKL_NUM_THREADS=str(c['threads']),
                       LOKY_MAX_CPU_COUNT=str(c['threads']), PYTORCH_ALLOC_CONF='expandable_segments:True')
            run(f'{model}/{name}/{scene}/adapt_all_seeds', item['command'],
                root / model / 'workers' / name / scene / 'worker.log', env)
            if a.skip_evaluation:
                continue
            available = [s for s in a.seeds if (root / model / f'seed_{s}' / name / scene / 'baseline/protocol.json').exists()]
            stages = ([(available[0], 'baseline')] if available else []) + ([(s, 'adapted') for s in a.seeds] if c.get('method') != 'baseline' else [])
            for seed, stage in stages:
                if name == 'dtu':
                    folder = root/model/f'seed_{seed}'
                    directory = folder/name/scene
                    if not (directory/stage/'exports/mini_npz/results.npz').exists():
                        continue
                    cmd = [sys.executable, str(ROOT/'scripts/run.py'), 'fuse',
                           '--config', str(root/model/f'config_seed_{seed}.json'),
                           '--output', str(folder), '--dataset', name, '--scene', scene, '--stage', stage]
                    if run(f'{model}/{name}/{scene}/seed_{seed}/{stage}_fuse', cmd,
                           directory/f'fuse_{stage}.log', env):
                        continue
                pending.append(pool.submit(evaluate, item, seed, stage, env))
        for future in pending:
            future.result()
    state.update(status='finished_with_failures' if failures else ('training_only' if a.skip_evaluation else 'complete'),
                 finished=time.time(), active=[])
    write_json(state_path, state)
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_full.py'), '--root', str(root)], check=False)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
