#!/usr/bin/env python3
"""Measure textureless and normalized-occlusion 8V subset feasibility.

This is analysis-only: it reads real RGB-D/pose artifacts and reports how many
of 128 candidate 8V windows per scene contain a Q4 hard student view.  7Scenes
is sampled every ten frames; HiRoom keeps every available real frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/covisibility_all_datasets_0025"
STUDENT = [0, 2, 4, 6]


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def rgb_path(dataset, depth_path):
    depth = Path(depth_path)
    if dataset == "7scenes":
        return depth.with_name(depth.name.replace(".depth.png", ".color.png"))
    image_dir = depth.parents[1] / "image"
    candidates = sorted(image_dir.glob(depth.stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"No RGB for {depth}")
    return candidates[0]


def depth_and_valid(dataset, path):
    path = Path(path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if dataset == "7scenes":
        depth = raw.astype(np.float32) / 1000.0
        valid = (raw != 65535) & (depth > 0)
    else:
        depth = raw.astype(np.float32) / 65535.0 * 100.0
        valid = depth > 0
        alias = path.parents[1] / "aliasing_mask" / path.name
        if alias.exists():
            valid &= cv2.imread(str(alias), cv2.IMREAD_UNCHANGED) == 0
    return depth, valid


def resize(depth, valid, intrinsic, max_side=224):
    h, w = depth.shape
    scale = min(1.0, max_side / max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(valid.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
    k = intrinsic.copy(); k[0] *= nw / w; k[1] *= nh / h; k[2, 2] = 1
    return depth, valid, k


def texture_scores(paths, device):
    values = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None: raise FileNotFoundError(path)
        image = cv2.resize(image, (504, 504), interpolation=cv2.INTER_AREA)
        values.append(image)
    x = torch.from_numpy(np.stack(values).astype(np.float32) / 255).unsqueeze(1).to(device)
    sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device).view(1,1,3,3)
    sy = sx.transpose(-1, -2)
    mag = torch.sqrt(F.conv2d(x, sx, padding=1).square() + F.conv2d(x, sy, padding=1).square())
    return F.avg_pool2d(mag, 14, 14).flatten(1).cpu().numpy()


@torch.inference_mode()
def occlusion_scores(depths, valids, intrinsics, w2cs, sequences, device):
    depths = torch.from_numpy(depths).to(device); valids = torch.from_numpy(valids).to(device)
    intrinsics = torch.from_numpy(intrinsics).to(device); w2cs = torch.from_numpy(w2cs).to(device)
    h, w = depths.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    pixels = torch.stack((xx.flatten(), yy.flatten(), torch.ones(h*w, device=device)))
    out=[]
    for sequence in sequences:
        row=[]
        for pos, source in enumerate(sequence):
            targets=[x for n,x in enumerate(sequence) if n != pos]
            z=depths[source].flatten()
            xyz=torch.linalg.solve(intrinsics[source], pixels)*z
            world=torch.linalg.inv(w2cs[source]) @ torch.cat((xyz, torch.ones(1,h*w,device=device)))
            cam=w2cs[targets] @ world.unsqueeze(0); pz=cam[:,2]
            uv=torch.bmm(intrinsics[targets], cam[:,:3]); u,v=uv[:,0]/uv[:,2],uv[:,1]/uv[:,2]
            inside=(pz>0)&(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
            grid=torch.stack((2*u/max(w-1,1)-1,2*v/max(h-1,1)-1),-1).view(7,h,w,2)
            td=F.grid_sample(depths[targets,None],grid,mode="nearest",padding_mode="zeros",align_corners=True)[:,0].reshape(7,-1)
            tv=F.grid_sample(valids[targets,None].float(),grid,mode="nearest",padding_mode="zeros",align_corners=True)[:,0].reshape(7,-1)>0.5
            candidate=inside&tv; test=valids[source].flatten()&(candidate.sum(0)>=2)
            seen=candidate&((td-pz).abs()<=.02*pz)
            row.append(float((test&(seen.sum(0)<2)).sum().float()/test.sum().clamp_min(1)))
        out.append(row)
    return np.asarray(out)


def candidates(n, dataset, scene, count=128):
    rng=np.random.default_rng(seed(dataset,scene,"hard-feasibility",30))
    ids=list(range(n)); out=[]
    for _ in range(count):
        out.append(rng.choice(ids, 8, replace=n<8).tolist())
    return out


def balanced(values, percentile):
    n=min(len(x) for x in values.values())
    return float(np.percentile(np.concatenate([x[np.linspace(0,len(x)-1,n,dtype=int)] for x in values.values()]),percentile))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--candidate-count",type=int,default=128)
    ap.add_argument("--output",type=Path,default=ROOT/"artifacts/hard_subset_feasibility_7scenes_hiroom.json")
    args=ap.parse_args(); device=args.device
    scenes={}; patches={}; sequence_map={}; geom={}
    for dataset in ("7scenes","hiroom"):
        for path in sorted((ARTIFACT/dataset).glob("*.npz")):
            with np.load(path,allow_pickle=False) as d:
                take=np.arange(0,len(d['frame_ids']),10) if dataset=="7scenes" else np.arange(len(d['frame_ids']))
                ids=[str(x) for x in d['frame_ids'][take]]; dp=[str(x) for x in d['depth_paths'][take]]
                paths=[rgb_path(dataset,x) for x in dp]
                key=f"{dataset}/{path.stem}"; scenes[key]=(dataset,path.stem,ids,dp,paths,np.asarray(d['intrinsics'][take]),np.asarray(d['extrinsics_w2c'][take]))
                patches[key]=texture_scores(paths,device); sequence_map[key]=candidates(len(ids),dataset,path.stem,args.candidate_count)
                print(f"[loaded] {key}: {len(ids)} candidate frames",flush=True)
    patch_q30={}
    for dataset in ("7scenes","hiroom"):
        subset={k:v.reshape(-1) for k,v in patches.items() if k.startswith(dataset+'/')}
        patch_q30[dataset]=balanced(subset,30)
    texture_frame={k:(v<patch_q30[k.split('/')[0]]).mean(1) for k,v in patches.items()}
    texture_q4={dataset:balanced({k:v for k,v in texture_frame.items() if k.startswith(dataset+'/')},75) for dataset in ("7scenes","hiroom")}
    result={"protocol":{"seven_scenes_stride":10,"hiroom_stride":1,"candidate_8v_windows":args.candidate_count,"student_positions":STUDENT,"normalized_occlusion":"candidate target: inside plus valid depth; denominator: at least two candidates; occluded: fewer than two 2%-depth-consistent observations"},"texture":{},"normalized_occlusion":{}}
    # Texture feasibility is immediately available from real RGB scores.
    for key, (dataset,scene,ids,dp,paths,k,w2c) in scenes.items():
        seqs=sequence_map[key]; scores=texture_frame[key]; threshold=texture_q4[dataset]
        eligible=[i for i, s in enumerate(seqs) if np.any(scores[s][STUDENT] >= threshold)]
        result['texture'][key]={"candidate_frames":len(ids),"frame_q4_threshold":threshold,"q4_frames":int((scores>=threshold).sum()),"eligible_8v_sequences":len(eligible),"eligible_candidate_indices":eligible,"feasible_5":len(eligible)>=5,"feasible_10":len(eligible)>=10}
    # Stream geometric data scene-by-scene to keep GPU memory bounded.
    occlusion_values={}
    for key,(dataset,scene,ids,dp,paths,k,w2c) in scenes.items():
        prepared=[resize(*depth_and_valid(dataset,x), intrinsic) for x,intrinsic in zip(dp,k)]
        depths=np.stack([x[0] for x in prepared]).astype(np.float32); valid=np.stack([x[1] for x in prepared]); ki=np.stack([x[2] for x in prepared])
        occlusion_values[key]=occlusion_scores(depths,valid,ki,w2c,sequence_map[key],device)
        print(f"[occlusion] {key}: {args.candidate_count} windows",flush=True)
        del depths,valid,ki
        torch.cuda.empty_cache()
    for dataset in ("7scenes","hiroom"):
        values={k:v.reshape(-1) for k,v in occlusion_values.items() if k.startswith(dataset+'/')}
        q4=balanced(values,75)
        for key,v in occlusion_values.items():
            if not key.startswith(dataset+'/'): continue
            eligible=np.flatnonzero((v[:,STUDENT]>=q4).any(1)).astype(int).tolist()
            result['normalized_occlusion'][key]={"candidate_frames":len(scenes[key][2]),"frame_position_q4_threshold":q4,"q4_student_instances":int((v[:,STUDENT]>=q4).sum()),"eligible_8v_sequences":len(eligible),"eligible_candidate_indices":eligible,"feasible_5":len(eligible)>=5,"feasible_10":len(eligible)>=10}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(args.output)

if __name__=='__main__': main()
