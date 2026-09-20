#!/usr/bin/env python3
"""Fixed diagnostic scenes: training curves, depth comparison and projected fused clouds."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from self_geometry import ROOT
from self_geometry.common import config
from self_geometry.data import dataset
from self_geometry.model import load_images

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'artifacts/main');a=p.parse_args()
for name in ['eth3d','7scenes','scannetpp','hiroom']:
    scene=dataset(name,config()).SCENES[0];d=a.root/name/scene
    if not (d/'adapted/complete.json').exists():continue
    out=d/'figures';out.mkdir(exist_ok=True)
    with np.load(d/'baseline/exports/mini_npz/results.npz') as z:base=z['depth']
    with np.load(d/'adapted/exports/mini_npz/results.npz') as z:final=z['depth']
    manifest=json.loads((d/'manifest.json').read_text())
    size=json.loads((d/'adapted/complete.json').read_text())['config']['image_size']
    rgb=load_images(manifest['image_files'][:1],size)[0].permute(1,2,0).numpy()
    vmax=np.percentile(base[0],98)
    fig,ax=plt.subplots(1,4,figsize=(14,4),constrained_layout=True)
    ax[0].imshow(rgb);ax[1].imshow(base[0],vmin=0,vmax=vmax,cmap='magma');ax[2].imshow(final[0],vmin=0,vmax=vmax,cmap='magma')
    delta=abs(final[0]-base[0]);ax[3].imshow(delta,vmin=0,vmax=max(np.percentile(delta,98),1e-6),cmap='inferno')
    for axis,title in zip(ax,['RGB (first evaluation frame)','Baseline depth','Adapted depth','Absolute prediction difference (not GT error)']):axis.set_title(title,fontsize=9);axis.axis('off')
    fig.suptitle(name+'/'+scene);fig.savefig(out/'depth_comparison.png',dpi=160);plt.close(fig)
    steps=[json.loads(line) for line in (d/'adapted/steps.jsonl').read_text().splitlines()]
    steps=[s for s in steps if 'losses' in s]
    fig,ax=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
    for loss in ['mvc','ec','pc','eds','bdc']:ax[0].plot([s['iteration'] for s in steps],[s['losses'][loss] for s in steps],label=loss)
    ax[0].set_yscale('symlog',linthresh=1e-5);ax[0].legend();ax[0].set_title('Robust losses')
    ax[1].plot([s['iteration'] for s in steps],[s['gradient_cosine'] for s in steps]);ax[1].axhline(0,color='gray',lw=.5);ax[1].set_title('MVC / EC gradient cosine')
    ax[2].plot([s['iteration'] for s in steps],[s['score'] for s in steps],label='current');ax[2].plot([s['iteration'] for s in steps],[s['best'] for s in steps],label='best');ax[2].legend();ax[2].set_title('Checkpoint criterion')
    for axis in ax:axis.set_xlabel('iteration')
    fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    paths=[d/stage/'exports/recon_unposed/pcd.ply' for stage in ['baseline','adapted']]
    if all(p.exists() for p in paths):
        import open3d as o3d
        clouds=[]
        for path in paths:
            xyz=np.asarray(o3d.io.read_point_cloud(str(path)).points)
            rng=np.random.default_rng(0);xyz=xyz[rng.choice(len(xyz),min(40000,len(xyz)),replace=False)]
            clouds.append(xyz)
        all_points=np.concatenate(clouds);lo=np.percentile(all_points,[1],axis=0)[0];hi=np.percentile(all_points,[99],axis=0)[0]
        fig,ax=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
        for axis,xyz,title in zip(ax,clouds,['Baseline unposed cloud','Adapted unposed cloud']):
            axis.scatter(xyz[:,0],xyz[:,2],s=.15,c=xyz[:,1],cmap='viridis',vmin=lo[1],vmax=hi[1],rasterized=True)
            axis.set_xlim(lo[0],hi[0]);axis.set_ylim(lo[2],hi[2]);axis.set_aspect('equal');axis.set_title(title);axis.set_xlabel('world x');axis.set_ylabel('world z')
        fig.savefig(out/'pointcloud_comparison.png',dpi=160);plt.close(fig)
    print(out)
