#!/usr/bin/env python3
"""Enable group OOM handling inside a delegated scope, then execute a worker."""
import json
import os
from pathlib import Path
import sys

group = next(line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
             if line.startswith('0::'))
if '/self-geometry.slice/' not in group:
    raise RuntimeError('Worker is outside the RAM-limited experiment slice')
base = Path('/sys/fs/cgroup') / group.lstrip('/')
limit = (base/'memory.max').read_text().strip()
if limit == 'max' or int(limit) > 72 * 1024**3:
    raise RuntimeError('Missing or excessive worker RAM limit')
# systemd 249 accepts MemoryMax but not the newer MemoryOOMGroup property.
# The delegated cgroup v2 interface supports it directly on this machine.
(base/'memory.oom.group').write_text('1')
if (base/'memory.oom.group').read_text().strip() != '1':
    raise RuntimeError('Could not enable OOM termination of the entire worker group')
print(json.dumps(dict(event='worker_ram_limit', bytes=int(limit), cgroup=group,
                      oom_group=True)), flush=True)
if len(sys.argv) < 2:
    raise RuntimeError('Missing worker command')
os.execvp(sys.argv[1], sys.argv[1:])
