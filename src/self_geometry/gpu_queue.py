"""Cross-process GPU admission for the local 96 GiB validation machine."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time

from . import ROOT
from .common import write_json

CAPACITY_GIB = 90


def reservation(c, baseline_protocol=None):
    # Unknown/high-memory schedules take the whole GPU. Test3R pair training
    # has measured peaks of 31.4/27.0 GiB at 336x504 (DA3/VGGT). Reserve
    # additional allocator headroom and scale conservatively with pixel count.
    if (c.get('method') != 'test3r' or c.get('test3r_pair_batch', 2) > 2
            or c.get('max_frames', 100) > 100 or baseline_protocol is None
            or c.get('precision', 'bf16') != 'bf16'
            or c.get('test3r_prompt_size', 32) > 32):
        return CAPACITY_GIB
    shape = baseline_protocol.get('shape', [])
    if len(shape) != 4 or c.get('image_size') != 504:
        return CAPACITY_GIB
    reference = 42 if c['model'] == 'da3' else 38
    return min(CAPACITY_GIB, math.ceil(reference * max(1., shape[-2]*shape[-1]/(378*504))))


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def reserve_gpu(gib=CAPACITY_GIB, path=None):
    """FIFO admission; stale reservations are removed after worker exit."""
    path = Path(path) if path else ROOT/'artifacts/gpu_admission.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = str(os.getpid())
    acquired = False
    announced = False
    try:
        while not acquired:
            with path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = json.loads(path.read_text()) if path.exists() else {}
                state = {p: v for p, v in state.items() if alive(p)}
                state.setdefault(pid, dict(gib=gib, status='waiting', since=time.time()))
                waiting = [p for p, v in state.items() if v['status'] == 'waiting']
                used = sum(v['gib'] for v in state.values() if v['status'] == 'running')
                if waiting[0] == pid and used + gib <= CAPACITY_GIB:
                    state[pid]['status'] = 'running'
                    acquired = True
                write_json(path, state)
            if not announced or acquired:
                print(json.dumps(dict(event='gpu_admission', pid=os.getpid(),
                                      gib=gib, status='running' if acquired else 'waiting')), flush=True)
                announced = True
            if not acquired:
                time.sleep(2)
        yield
    finally:
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(path.read_text()) if path.exists() else {}
            state.pop(pid, None)
            write_json(path, state)
