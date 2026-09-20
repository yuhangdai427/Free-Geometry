import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from . import ROOT


def config(path=None, overrides=(), model=None):
    c = yaml.safe_load((ROOT / 'configs/paper.yaml').read_text())
    if path:
        path = Path(path)
        c.update((json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())) or {})
    if model is not None:
        c['model'] = model
        c['weights'] = '/localhdd02/yuhang/weights/' + ('DA3-GIANT-1.1/model.safetensors' if model == 'da3' else 'VGGT-1B/model.pt')
    for item in overrides:
        k, v = item.split('=', 1)
        if k not in c:
            raise ValueError(f'Unknown setting: {k}')
        try:
            c[k] = json.loads(v)
        except json.JSONDecodeError:
            c[k] = yaml.safe_load(v)
    for key in ('lr', 'final_lr', 'warmup_fraction', 'weight_decay', 'clip', 'alpha', 'dropout', 'filter_keep'):
        c[key] = float(c[key])
    if not isinstance(c['checkpoint_every'], int) or c['checkpoint_every'] < 1:
        raise ValueError('checkpoint_every must be a positive integer')
    return c


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp); temp.replace(path)


def seed_all(seed, threads=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])


@contextmanager
def isolated_rng():
    """Preprocessing/cache construction must not advance the adaptation RNG."""
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)
