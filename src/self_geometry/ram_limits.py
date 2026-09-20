"""Kernel-enforced RAM limits shared across experiment runners (cgroup v2)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from . import ROOT

SLICE = 'self-geometry.slice'
TOTAL_GIB = 72
HIGH_GIB = 64
WORKER_GIB = 32


def prepare_slice():
    # A user-owned systemd slice gives all concurrent matrices one aggregate
    # ceiling; per-process RLIMIT_AS would incorrectly count CUDA address space.
    check = subprocess.run(['systemctl', '--user', 'show', SLICE, '-p', 'LoadState', '--value'],
                           capture_output=True, text=True)
    if check.returncode or check.stdout.strip() != 'loaded':
        subprocess.run(['systemd-run', '--user', '--scope', '--quiet', f'--slice={SLICE}',
                        '/bin/true'], check=True)
    subprocess.run(['systemctl', '--user', 'set-property', '--runtime', SLICE,
                    f'MemoryMax={TOTAL_GIB}G', f'MemoryHigh={HIGH_GIB}G', 'MemorySwapMax=0'], check=True)
    group = subprocess.check_output(['systemctl', '--user', 'show', SLICE,
                                     '-p', 'ControlGroup', '--value'], text=True).strip()
    base = Path('/sys/fs/cgroup') / group.lstrip('/')
    if int((base/'memory.max').read_text()) != TOTAL_GIB * 1024**3:
        raise RuntimeError('RAM hard limit was not applied; refusing to launch unprotected work')
    if int((base/'memory.swap.max').read_text()) != 0:
        raise RuntimeError('Task swap limit was not applied')
    return base


def scope_command(command, limit_gib=WORKER_GIB):
    if not 0 < limit_gib <= TOTAL_GIB:
        raise ValueError('Worker RAM limit must be positive and no greater than the shared limit')
    return ['systemd-run', '--user', '--scope', '--quiet',
            f'--unit=self-geometry-job-{uuid.uuid4().hex}', f'--slice={SLICE}',
            f'--property=MemoryMax={limit_gib}G', '--property=MemorySwapMax=0',
            '--', '/usr/bin/python3', str(ROOT/'scripts/cgroup_exec.py'), *command]


def enter_runner_scope():
    prepare_slice()
    if f'/{SLICE}/' in Path('/proc/self/cgroup').read_text():
        return
    print(json.dumps(dict(event='host_ram_limit', shared_gib=TOTAL_GIB,
                          high_gib=HIGH_GIB, worker_gib=WORKER_GIB, swap_gib=0)), flush=True)
    cmd = scope_command([sys.executable, *sys.argv], TOTAL_GIB)
    os.execvp(cmd[0], cmd)
