#!/usr/bin/env python3
"""Run a comparison command after another matrix releases its worker processes."""
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
from self_geometry.common import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('Provide the command after --')
    root, parent = a.root.resolve(), a.after.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='queued', pid=os.getpid(), after=str(parent), command=command,
                 queued=time.time())
    write_json(root/'queue_state.json', state)
    print(json.dumps(state), flush=True)
    while True:
        before = json.loads((parent/'matrix_state.json').read_text())
        if before['status'] != 'running':
            # Terminal state is written only after all training/eval subprocesses
            # return. A failed scene must not cancel the remaining comparison.
            break
        try:
            os.kill(before['pid'], 0)
        except ProcessLookupError:
            state.update(status='blocked', error='Parent disappeared before writing terminal state')
            write_json(root/'queue_state.json', state)
            return 1
        time.sleep(15)
    state.update(status='running', started=time.time(), parent_status=before['status'])
    write_json(root/'queue_state.json', state)
    print(json.dumps(state), flush=True)
    code = subprocess.run(command, cwd=ROOT).returncode
    state.update(status='complete' if code == 0 else 'finished_with_failures',
                 exit_code=code, finished=time.time())
    write_json(root/'queue_state.json', state)
    return code


if __name__ == '__main__':
    sys.exit(main())
