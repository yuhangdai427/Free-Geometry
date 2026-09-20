#!/usr/bin/env python3
"""Optional ambiguity experiments; never part of the default full benchmark."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from self_geometry import ROOT
from self_geometry.common import config,write_json


from self_geometry.cache import reuse


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'artifacts/diagnostics');a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    schedule=[]
    variants=['resolution518','filter95','initial_filter','reverse_gd','no_gd','no_fan','bins9','conflict_only','seed1','seed2']
    schedule += [(f'variants/{v}',ROOT/f'configs/variants/{v}.yaml',True,None) for v in variants]
    schedule += [(f'variants/{v}',ROOT/f'configs/variants/{v}.yaml',False,['eth3d']) for v in ['no_gd','no_fan']]
    records=[]
    for name,path,first,datasets in schedule:
        dest=root/name;c=config(path)
        if name!='main':reuse(ROOT/'artifacts/main',dest,c)
        cmd=[sys.executable,str(ROOT/'scripts/campaign.py'),'--output',str(dest)]
        if path:cmd+=['--config',str(path)]
        if first:cmd+=['--first-only']
        cmd+=['--datasets']+(datasets or ['eth3d','7scenes','scannetpp','hiroom'])
        tick=time.time();print('EXPERIMENT',name,'first_only',first,flush=True)
        ret=subprocess.run(cmd).returncode
        records.append(dict(name=name,first_only=first,datasets=datasets,exit_code=ret,seconds=time.time()-tick))
        write_json(root/'experiments.json',records)
    subprocess.run([sys.executable,str(ROOT/'scripts/diagnostics/summarize_experiments.py'),'--root',str(root)],check=False)
    subprocess.run([sys.executable,str(ROOT/'scripts/visualize.py'),'--root',str(root/'main')],check=False)
    return int(any(r['exit_code'] for r in records))

if __name__=='__main__':sys.exit(main())
