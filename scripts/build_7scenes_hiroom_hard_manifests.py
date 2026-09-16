#!/usr/bin/env python3
"""Materialize analysis-approved 8V/16V manifests for 7Scenes and HiRoom."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/covisibility_all_datasets_0025"
FEASIBILITY = ROOT / "artifacts/hard_subset_feasibility_7scenes_hiroom.json"
OUT = ROOT / "artifacts/hard_view_subsets_7scenes_hiroom"
STUDENT = [0, 2, 4, 6]


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def image_path(dataset, depth):
    p = Path(depth)
    if dataset == "7scenes": return str(p.with_name(p.name.replace(".depth.png", ".color.png")))
    matches = sorted((p.parents[1] / "image").glob(f"{p.stem}.*"))
    if not matches: raise FileNotFoundError(p)
    return str(matches[0])


def load_scene(dataset, scene):
    p = SOURCE / dataset / f"{scene}.npz"
    with np.load(p, allow_pickle=False) as d:
        real_scene = str(d["scene"].item())
        take = np.arange(0, len(d["frame_ids"]), 10) if dataset == "7scenes" else np.arange(len(d["frame_ids"]))
        ids = [str(x) for x in d["frame_ids"][take]]; dp = [str(x) for x in d["depth_paths"][take]]
    return p, real_scene, ids, [image_path(dataset, x) for x in dp]


def sequences(n, count, dataset, scene, tag, views):
    rng=np.random.default_rng(stable_seed(dataset,scene,tag)); pool=list(range(n)); rows=[]
    for _ in range(count):
        if n >= views: rows.append(rng.choice(pool,views,replace=False).tolist())
        else: rows.append(pool.copy())
    return rows


def analysis_candidates(n, dataset, scene, count=128):
    rng=np.random.default_rng(stable_seed(dataset,scene,"hard-feasibility",30)); pool=list(range(n))
    return [rng.choice(pool, 8, replace=n<8).tolist() for _ in range(count)]


def record(dataset,scene,ids,images,seq,split,idx,epoch,criterion,views):
    four=[seq[i] for i in STUDENT] if views==8 else []
    out={"dataset":dataset,"scene":scene,"split":split,"sample_idx":idx,"epoch":epoch,
         "criterion":criterion,"difficulty_score":0.0,"hard_student_positions":[],"repeated_candidate":False}
    prefix="eight" if views==8 else "sixteen"
    out[f"{prefix}_local_indices"]=seq; out[f"{prefix}_frame_indices"]=seq
    out[f"{prefix}_frame_ids"]=[ids[i] for i in seq]; out[f"{prefix}_image_files"]=[images[i] for i in seq]
    if views==8:
        out["four_local_indices"]=four; out["four_frame_indices"]=four
        out["four_frame_ids"]=[ids[i] for i in four]; out["four_image_files"]=[images[i] for i in four]
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-root",type=Path,default=OUT); args=ap.parse_args()
    feasibility=json.loads(FEASIBILITY.read_text())
    for dataset in ("7scenes","hiroom"):
        dataset_dir=args.output_root/dataset; dataset_dir.mkdir(parents=True,exist_ok=True)
        source_records=[]; combined_train=[]
        for criterion, key in (("texture","texture"),("occlusion","normalized_occlusion")):
            rows=[]
            candidates=[(name.split('/',1)[1], value) for name,value in feasibility[key].items()
                        if name.startswith(dataset+'/') and value["feasible_10"]]
            # Same rule as prior subsets: highest number of eligible hard 8V windows.
            selected=[name for name,_ in sorted(candidates,key=lambda x:(-x[1]["eligible_8v_sequences"],x[0]))[:5]]
            train=[]; eval8=[]; eval16=[]
            for scene in selected:
                matrix,real_scene,ids,images=load_scene(dataset,scene)
                source_records.append({"dataset":dataset,"scene":real_scene,"total_frames":len(ids),"selected_count":len(ids),
                                       "selected_frame_indices":list(range(len(ids))),"selected_frame_ids":ids,"matrix_path":str(matrix)})
                hard_rows=[analysis_candidates(len(ids),dataset,scene)[i] for i in dict(candidates)[scene]["eligible_candidate_indices"]]
                for epoch in range(3):
                    for idx in range(10):
                        seq=hard_rows[idx % len(hard_rows)]
                        item=record(dataset,real_scene,ids,images,seq,"train",idx,epoch,criterion,8); train.append(item); combined_train.append(item)
                for idx in range(5):
                    seq=hard_rows[(10 + idx) % len(hard_rows)]
                    eval8.append(record(dataset,real_scene,ids,images,seq,"eval",idx,None,criterion,8))
                for idx,seq in enumerate(sequences(len(ids),5,dataset,scene,f"{criterion}-eval16",16)):
                    eval16.append(record(dataset,real_scene,ids,images,seq,"eval_16v",idx,None,criterion,16))
            root=dataset_dir/criterion; root.mkdir(parents=True,exist_ok=True)
            for name,items in (("train.jsonl",train),("eval.jsonl",eval8),("eval_16v.jsonl",eval16)):
                (root/name).write_text("".join(json.dumps(x)+"\n" for x in items))
            (root/"selection_report.json").write_text(json.dumps({"dataset":dataset,"criterion":criterion,"top5_scenes":selected,"train_records":len(train),"eval_8v_records":len(eval8),"eval_16v_records":len(eval16)},indent=2))
        # Deduplicate source records: a scene can be selected under both criteria.
        unique={(x['dataset'],x['scene']):x for x in source_records}
        (dataset_dir/"source_frames.jsonl").write_text("".join(json.dumps(x)+"\n" for x in unique.values()))
        (dataset_dir/"train_combined.jsonl").write_text("".join(json.dumps(x)+"\n" for x in combined_train))
        print(dataset, "combined_train",len(combined_train),"source_scenes",len(unique))

if __name__=="__main__": main()
