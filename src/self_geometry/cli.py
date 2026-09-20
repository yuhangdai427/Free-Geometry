import argparse
import json
import os
from pathlib import Path
from . import ROOT
from .common import config, write_json
from .benchmark import DATASETS


def main():
    parser=argparse.ArgumentParser(description='Independent Self-Geometry reproduction')
    parser.add_argument('command',choices=['prepare','baseline','adapt','evaluate','fuse','score','report'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--model',choices=['vggt','da3'])
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--scene')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/main')
    parser.add_argument('--stage',choices=['baseline','adapted'],default='adapted')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();c=config(args.config,args.set,model=args.model)
    os.environ['TORCH_HOME']=str(ROOT/'weights')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    args.output=args.output.resolve()
    os.chdir(ROOT)
    if args.command=='report':
        from .report import report
        report(args.output,c);return
    if not args.dataset: parser.error('--dataset is required')
    from .data import dataset,prepare,evaluate
    ds=dataset(args.dataset,c)
    scenes=[args.scene] if args.scene else ds.SCENES
    for scene in scenes:
        directory=args.output/args.dataset/scene
        if args.command=='prepare': prepare(args.dataset,scene,c,directory)
        elif args.command=='baseline':
            from .training import baseline
            prepare(args.dataset,scene,c,directory);baseline(c,directory)
        elif args.command=='adapt':
            if c.get('method', 'self_geometry') == 'self_geometry':
                from .training import adapt
                adapt(c,directory,args.resume)
            elif c.get('method') in ('test3r', 'tco'):
                from .comparisons import adapt_comparison
                adapt_comparison(c,directory,args.dataset,resume=args.resume)
            else:
                parser.error('baseline has no adaptation stage')
        else:
            print(json.dumps(evaluate(args.dataset,scene,c,directory,args.stage,
                                      fuse_only=args.command=='fuse',score_only=args.command=='score')),flush=True)

if __name__=='__main__':main()
