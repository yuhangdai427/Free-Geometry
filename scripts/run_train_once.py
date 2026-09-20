#!/usr/bin/env python3
"""One per-scene adaptation, then evaluation sampling seeds 42/43/44."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry import ROOT
from self_geometry.common import write_json
from self_geometry.ram_limits import enter_runner_scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/train_once_eval_three')
    p.add_argument('--train-seed', type=int, default=0)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--worker-ram-gib', type=int, default=32)
    p.add_argument('--first-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a, extra = p.parse_known_args()
    if any(x.split('=')[0] in ('--seeds', '--skip-evaluation') for x in extra):
        p.error('Use --train-seed for one training RNG and --eval-seeds for evaluation sampling')
    root = a.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train = ['bash', str(ROOT/'scripts/run_comparison.sh'), '--root', str(root/'training'),
             '--config', str(ROOT/'configs/comparison.yaml'), '--set', 'checkpointing=false',
             '--methods', 'baseline', 'self_geometry', 'tco', 'test3r', '--seeds', str(a.train_seed),
             '--protocol', 'paper', '--worker-ram-gib', str(a.worker_ram_gib), *extra]
    if a.first_only:
        train.append('--first-only')
    evaluate = [sys.executable, str(ROOT/'scripts/evaluate_seeds.py'), '--source', str(root/'training'),
                '--output', str(root/'evaluation'), '--eval-seeds', *map(str, a.eval_seeds),
                '--worker-ram-gib', str(a.worker_ram_gib)]
    plan = dict(train_seed=a.train_seed, train_frame_sampling_seed=42, evaluation_seeds=a.eval_seeds,
                adaptation='one independent adapter per model/method/scene',
                deduplication='same ordered RGB files and checkpoint evaluated once',
                training_command=train, evaluation_command=evaluate)
    path = root/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Different train/evaluation plan; choose another root')
    write_json(path, plan)
    if a.dry_run:
        return subprocess.run([*train, '--dry-run'], cwd=ROOT).returncode
    state = dict(pid=os.getpid(), started=time.time(), status='training_and_eval42')
    write_json(root/'state.json', state)
    code = subprocess.run(train, cwd=ROOT).returncode
    state.update(training_exit_code=code, status='evaluation_samples')
    write_json(root/'state.json', state)
    # Failed scenes do not prevent completed checkpoints from being evaluated.
    eval_code = subprocess.run(evaluate, cwd=ROOT).returncode
    state.update(evaluation_exit_code=eval_code, finished=time.time(),
                 status='finished_with_failures' if code or eval_code else 'complete')
    write_json(root/'state.json', state)
    return int(bool(code or eval_code))


if __name__ == '__main__':
    enter_runner_scope()
    sys.exit(main())
