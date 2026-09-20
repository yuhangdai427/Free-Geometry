#!/usr/bin/env python3
"""Fetch pinned author source code only; never downloads models or datasets."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry import ROOT
from self_geometry.comparisons import TCO_SHA, TEST3R_SHA
for name,url,sha in [('TCO','https://github.com/cvlab-stonybrook/TCO.git',TCO_SHA),
                     ('Test3R','https://github.com/nopQAQ/Test3R.git',TEST3R_SHA)]:
    path = ROOT/'external'/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',url,str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',sha],check=True)
    head = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if head != sha or subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():
        raise RuntimeError(f'{path}: expected clean pinned revision {sha}; refusing to overwrite')
    print(name,head)
