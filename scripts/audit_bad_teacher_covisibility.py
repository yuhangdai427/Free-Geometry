#!/usr/bin/env python3
"""Audit geometry of seeded windows where the 8V-to-4V teacher is worse.

The seed replay matches the original baseline sampler exactly: Python
``random.Random(seed).sample(range(num_frames), 8)``, sorted afterwards.  It
does not select views based on Co-visibility.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compute_covisibility_matrices import compute_matrix, get_frames, read_depth, resize_frame, DEFAULT_ROOTS


BAD = {
 "vggt": {
  "eth3d": {"electro":[43],"facade":[40,42,43],"kicker":[43],"office":[44],"pipes":[43],"playground":[40],"relief_2":[41]},
  "7scenes": {"chess":[41,43,44,45,46],"fire":[41,43,47,49],"heads":[40],"office":[42,48],"pumpkin":[40,41,49],"redkitchen":[42,44,45],"stairs":[42,43,46]},
  "scannetpp": {"09c1414f1b":[44,45,46,47],"1ada7a0617":[40,46],"21d970d8de":[41,46],"286b55a2bf":[41],"38d58a7a31":[48,49],"3e8bba0176":[48],"40aec5fffa":[43,45],"5f99900f09":[47,49],"7831862f02":[43,47,48,49],"7bc286c1b6":[47,48],"9071e139d9":[40,42,43,45,46,49],"acd95847c5":[40,41,45],"bcd2436daf":[43],"bde1e479ad":[44,49],"c4c04e6d6c":[41,42,44,45,46],"cc5237fd77":[40,43,47,48],"f3d64c30f8":[42,44,46,48],"fb5a96b1a2":[40,43,49]},
  "hiroom": {"20241230/828744/cam_sampled_04":[43],"20241230/828745/cam_sampled_05":[41],"20241230/828749/cam_sampled_12":[44],"20241230/828750/cam_sampled_08":[40,41,42,44],"20241230/828757/cam_sampled_06":[41,42,43],"20241230/828763/cam_sampled_12":[42],"20241230/828769/cam_sampled_11":[41],"20241230/828770/cam_sampled_06":[40],"20241230/828777/cam_sampled_10":[44],"20241230/828782/cam_sampled_09":[42,44],"20241230/828786/cam_sampled_10":[44],"20241230/828788/cam_sampled_13":[41,42],"20241230/828791/cam_sampled_12":[40],"20241230/828792/cam_sampled_10":[41],"20241230/828805/cam_sampled_06":[44],"20241230/828811/cam_sampled_02":[42],"20241230/828815/cam_sampled_04":[40,42],"20241230/828816/cam_sampled_14":[42,43],"20241230/828824/cam_sampled_08":[40,43],"20241230/828825/cam_sampled_07":[42]},
 },
 "da3": {
  "eth3d": {"courtyard":[41],"delivery_area":[42,43],"electro":[41],"facade":[43,44],"kicker":[41,43],"office":[40],"pipes":[40,42,43],"playground":[40,41,42],"relief_2":[41,42]},
  "7scenes": {"chess":[41,43,44,45,46,49],"fire":[43,48],"heads":[42,48],"office":[41,42,44,45],"pumpkin":[40,41,43,44,47,49],"redkitchen":[41,42,43],"stairs":[46]},
  "scannetpp": {"09c1414f1b":[45,46],"1ada7a0617":[40,41,42,44],"21d970d8de":[41,43],"286b55a2bf":[45],"38d58a7a31":[41,48],"3e8bba0176":[42,44],"40aec5fffa":[41,48],"578511c8a9":[47],"5f99900f09":[43,48],"7831862f02":[42,43,47],"7bc286c1b6":[40,41,45],"9071e139d9":[41,47,48],"acd95847c5":[45,46,49],"bcd2436daf":[40,43],"bde1e479ad":[43,44],"c4c04e6d6c":[41,45,48],"c5439f4607":[41,42,49],"f3d64c30f8":[47],"fb5a96b1a2":[45]},
  "hiroom": {"20241230/828738/cam_sampled_08":[43],"20241230/828744/cam_sampled_04":[44],"20241230/828749/cam_sampled_12":[40,44],"20241230/828750/cam_sampled_08":[42],"20241230/828763/cam_sampled_12":[41],"20241230/828766/cam_sampled_10":[42],"20241230/828770/cam_sampled_06":[42],"20241230/828774/cam_sampled_23":[42],"20241230/828782/cam_sampled_09":[44],"20241230/828784/cam_sampled_03":[40,41],"20241230/828785/cam_sampled_12":[40,44],"20241230/828786/cam_sampled_10":[40],"20241230/828794/cam_sampled_10":[40],"20241230/828816/cam_sampled_14":[44],"20241230/828824/cam_sampled_08":[40,43]},
 },
}

MATRIX_DIR = {"eth3d": ROOT / "artifacts/covisibility_eth3d_all/eth3d", "scannetpp": ROOT / "artifacts/covisibility_scannetpp_all_frames/scannetpp", "7scenes": ROOT / "artifacts/covisibility_all_datasets_0025/7scenes", "hiroom": ROOT / "artifacts/covisibility_all_datasets_0025/hiroom"}

def matrix_path(dataset: str, scene: str) -> Path:
    return MATRIX_DIR[dataset] / f"{scene.replace('/', '__')}.npz"

def load_full(dataset: str, scene: str, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return full-frame overlap matrix and original frame indices."""
    if dataset != "7scenes":
        with np.load(matrix_path(dataset, scene), allow_pickle=False) as d:
            return np.asarray(d["max_directional_overlap"]), np.asarray(d["candidate_indices"])
    # Existing 7Scenes artifacts are stride-10; recompute only the requested
    # exact eight-frame windows below, while its scene reference uses artifact pairs.
    with np.load(ROOT / "artifacts/covisibility_all_datasets_0025/7scenes" / f"{scene}.npz", allow_pickle=False) as d:
        return np.asarray(d["max_directional_overlap"]), np.asarray(d["candidate_indices"])

def exact_7scene(scene: str, indices: list[int], device: str) -> np.ndarray:
    frames = get_frames("7scenes", DEFAULT_ROOTS["7scenes"], scene)
    prepared = [resize_frame(*read_depth(frames[i], "7scenes"), 224) for i in indices]
    depth, valid, intrinsic = (np.stack(x) for x in zip(*prepared))
    raw = compute_matrix(depth, valid, intrinsic, np.stack([frames[i].w2c for i in indices]), device, 8)
    directional = raw / (np.diag(raw)[:, None] + 1e-8)
    return np.maximum(directional, directional.T)

def stats(overlap: np.ndarray) -> dict:
    tri = overlap[np.triu_indices(len(overlap), 1)]
    pairs = [(i, j) for i in range(len(overlap)) for j in range(i + 1, len(overlap))]
    extra = {1, 3, 5, 7}
    extra_pair_values = np.asarray([overlap[i, j] for i, j in pairs if i in extra or j in extra])
    # Treat CoVis as a weighted graph.  The maximum spanning-tree bottleneck is
    # the strongest possible weakest edge needed to connect all eight views.
    # Low values identify a view that can join the whole context only weakly.
    def max_tree_bottleneck() -> float:
        """Kruskal maximum spanning tree, including genuine zero-weight edges."""
        parent = list(range(len(overlap)))
        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]; node = parent[node]
            return node
        selected = []
        for weight, left, right in sorted(((float(overlap[i, j]), i, j) for i, j in pairs), reverse=True):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_left] = root_right; selected.append(weight)
                if len(selected) == len(overlap) - 1: break
        return min(selected) if selected else 0.0
    tree_bottleneck = max_tree_bottleneck()
    degree_010 = (overlap >= 0.1).sum(axis=1) - 1
    degree_020 = (overlap >= 0.2).sum(axis=1) - 1
    node_mean = (overlap.sum(axis=1) - np.diag(overlap)) / (len(overlap) - 1)
    # Number of connected components at each threshold, via a tiny DFS.
    def components(threshold: float) -> int:
        adjacent = overlap >= threshold
        seen, count = set(), 0
        for start in range(len(overlap)):
            if start in seen: continue
            count += 1; stack = [start]
            while stack:
                node = stack.pop()
                if node in seen: continue
                seen.add(node)
                stack.extend(np.flatnonzero(adjacent[node]).tolist())
        return count
    student_stats = None
    if len(overlap) == 8:
        student_overlap = overlap[np.ix_([0, 2, 4, 6], [0, 2, 4, 6])]
        student_stats = stats(student_overlap)
    result = {
        "mean_pair_overlap":float(tri.mean()), "median_pair_overlap":float(np.median(tri)),
        "min_pair_overlap":float(tri.min()), "max_pair_overlap":float(tri.max()),
        "fraction_pairs_le_010":float((tri<=.1).mean()), "fraction_pairs_le_020":float((tri<=.2).mean()),
        "has_any_pair_le_010":bool((tri<=.1).any()), "has_any_pair_le_020":bool((tri<=.2).any()),
        "min_node_mean_overlap":float(node_mean.min()),
        "max_spanning_tree_bottleneck":float(tree_bottleneck),
        "components_at_010":components(0.1), "components_at_020":components(0.2),
        "isolated_nodes_at_010":int((degree_010 == 0).sum()), "isolated_nodes_at_020":int((degree_020 == 0).sum()),
    }
    if len(overlap) == 8:
        # The teacher-only context is positions 1/3/5/7.  This distinguishes
        # a weak context-to-student link from a weak pair wholly inside 4V.
        result.update({
            "min_pair_involving_extra":float(extra_pair_values.min()),
            "has_extra_pair_le_010":bool((extra_pair_values<=.1).any()),
            "has_extra_pair_le_020":bool((extra_pair_values<=.2).any()),
            "min_extra_to_student_mean_overlap":float(min(overlap[position, [0, 2, 4, 6]].mean() for position in extra)),
            "student_components_at_010": student_stats["components_at_010"],
            "student_components_at_020": student_stats["components_at_020"],
            "student_mst_bottleneck": student_stats["max_spanning_tree_bottleneck"],
            "student_isolated_at_010": student_stats["isolated_nodes_at_010"],
            "extra_causes_disconnection_at_010": bool(student_stats["components_at_010"] == 1 and components(0.1) > 1),
            "extra_causes_disconnection_at_020": bool(student_stats["components_at_020"] == 1 and components(0.2) > 1),
            "extra_lowers_mst_bottleneck": float(student_stats["max_spanning_tree_bottleneck"] - tree_bottleneck),
        })
    return result

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0");p.add_argument("--output",default="artifacts/bad_teacher_covisibility_audit.json"); args=p.parse_args()
    rows=[]; cache={}
    for family, datasets in BAD.items():
      for dataset, scenes in datasets.items():
       for scene,seeds in scenes.items():
        key=(dataset,scene)
        if key not in cache: cache[key]=load_full(dataset,scene,args.device)
        reference, candidate_indices=cache[key]
        scene_stats=stats(reference)
        lookup={int(v):i for i,v in enumerate(candidate_indices)}
        # Candidate/frame order exactly follows the benchmark loader for these matrices.
        for seed in seeds:
          num_frames = int(candidate_indices.max())+1 if dataset!="7scenes" else (500 if scene=="stairs" else 1000)
          indices=sorted(random.Random(seed).sample(range(num_frames),8))
          if dataset=="7scenes": overlap=exact_7scene(scene,indices,args.device)
          else: overlap=reference[np.ix_([lookup[i] for i in indices],[lookup[i] for i in indices])]
          row={"model_family":family,"dataset":dataset,"scene":scene,"seed":seed,"frame_indices":indices,**stats(overlap),"scene_mean_pair_overlap":scene_stats["mean_pair_overlap"],"scene_pair_percentile":float((reference[np.triu_indices(len(reference),1)]<=stats(overlap)["mean_pair_overlap"]).mean())}
          row["mean_minus_scene_mean"]=row["mean_pair_overlap"]-row["scene_mean_pair_overlap"]
          rows.append(row)
    summary={}
    for family in BAD:
      summary[family]={}
      for dataset in BAD[family]:
       values=[r for r in rows if r['model_family']==family and r['dataset']==dataset]
       summary[family][dataset]={"windows":len(values),"mean_window_overlap":float(np.mean([r['mean_pair_overlap'] for r in values])),"mean_scene_overlap":float(np.mean([r['scene_mean_pair_overlap'] for r in values])),"mean_delta":float(np.mean([r['mean_minus_scene_mean'] for r in values])),"below_scene_mean_rate":float(np.mean([r['mean_minus_scene_mean']<0 for r in values])),"mean_scene_percentile":float(np.mean([r['scene_pair_percentile'] for r in values])),"strict_all_pairs_le_010_rate":float(np.mean([r['max_pair_overlap']<=.1 for r in values])),"any_pair_le_010_rate":float(np.mean([r['has_any_pair_le_010'] for r in values])),"any_pair_le_020_rate":float(np.mean([r['has_any_pair_le_020'] for r in values])),"any_extra_pair_le_010_rate":float(np.mean([r['has_extra_pair_le_010'] for r in values])),"any_extra_pair_le_020_rate":float(np.mean([r['has_extra_pair_le_020'] for r in values])),"mean_min_pair_overlap":float(np.mean([r['min_pair_overlap'] for r in values])),"mean_min_node_overlap":float(np.mean([r['min_node_mean_overlap'] for r in values])),"mean_extra_to_student_overlap":float(np.mean([r['min_extra_to_student_mean_overlap'] for r in values])),"mean_mst_bottleneck":float(np.mean([r['max_spanning_tree_bottleneck'] for r in values])),"disconnected_at_010_rate":float(np.mean([r['components_at_010']>1 for r in values])),"disconnected_at_020_rate":float(np.mean([r['components_at_020']>1 for r in values])),"mean_isolated_at_010":float(np.mean([r['isolated_nodes_at_010'] for r in values])),"student_connected_at_010_rate":float(np.mean([r['student_components_at_010']==1 for r in values])),"extra_causes_disconnection_at_010_rate":float(np.mean([r['extra_causes_disconnection_at_010'] for r in values])),"extra_causes_disconnection_at_020_rate":float(np.mean([r['extra_causes_disconnection_at_020'] for r in values])),"mean_extra_mst_drop":float(np.mean([r['extra_lowers_mst_bottleneck'] for r in values]))}
    # Matched random control: identical baseline sampler, every scene with an
    # available matrix, seeds 40--49. This tells us whether merely having one
    # weak pair is exceptional or routine under ordinary 8V sampling.
    controls = defaultdict(list)
    for dataset, directory in MATRIX_DIR.items():
      for path in sorted(directory.glob("*.npz")):
       scene = path.stem.replace("__", "/") if dataset == "hiroom" else path.stem
       reference, candidate_indices = load_full(dataset, scene, args.device)
       lookup = {int(value): position for position, value in enumerate(candidate_indices)}
       count = int(candidate_indices.max()) + 1 if dataset != "7scenes" else (500 if scene == "stairs" else 1000)
       for seed in range(40, 50):
        indices = sorted(random.Random(seed).sample(range(count), 8))
        overlap = exact_7scene(scene, indices, args.device) if dataset == "7scenes" else reference[np.ix_([lookup[i] for i in indices], [lookup[i] for i in indices])]
        controls[dataset].append(stats(overlap))
    control_summary = {dataset: {"windows": len(values), **{key: float(np.mean([item[key] for item in values])) for key in (
        "has_any_pair_le_010", "has_any_pair_le_020", "has_extra_pair_le_010", "has_extra_pair_le_020", "min_pair_overlap",
        "min_node_mean_overlap", "min_extra_to_student_mean_overlap", "max_spanning_tree_bottleneck",
        "components_at_010", "components_at_020", "isolated_nodes_at_010", "student_components_at_010",
        "student_components_at_020", "student_mst_bottleneck", "extra_causes_disconnection_at_010",
        "extra_causes_disconnection_at_020", "extra_lowers_mst_bottleneck",
    )}} for dataset, values in controls.items()}
    out={"sampling":"sorted(random.Random(seed).sample(range(num_frames), 8)); student = positions [0,2,4,6]","rows":rows,"summary":summary,"matched_random_controls":control_summary}
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
