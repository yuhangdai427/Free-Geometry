#!/usr/bin/env python3
"""Prioritize three already-planned controls for the failing HiRoom diagnostic."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from self_geometry import ROOT
from self_geometry.common import config
from self_geometry.cache import reuse

variants=['initial_filter','reverse_gd','no_gd']
for name in variants:
    reuse(ROOT/'artifacts/main',ROOT/f'artifacts/diagnostics/variants/{name}',config(ROOT/f'configs/variants/{name}.yaml'))
def run(name):
    cmd=[sys.executable,str(ROOT/'scripts/campaign.py'),'--output',str(ROOT/f'artifacts/diagnostics/variants/{name}'),
         '--config',str(ROOT/f'configs/variants/{name}.yaml'),'--datasets','hiroom','--first-only']
    return subprocess.run(cmd,cwd=ROOT).returncode
with ThreadPoolExecutor(max_workers=2) as pool:
    codes=list(pool.map(run,variants))
subprocess.run([sys.executable,str(ROOT/'scripts/diagnostics/summarize_experiments.py')],cwd=ROOT)
sys.exit(int(any(codes)))
