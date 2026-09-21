#!/usr/bin/env python3
"""Build N-dispatch manifests (2026-09-20): 5 datasets x (train/probe pairs via
protocol_v1.build_scene_protocol with the N>=64->16:4 else 8:4 auto rule).

- 7scenes/eth3d/hiroom/scannetpp: scene list + eval32_frames inherited VERBATIM
  from artifacts/diagnostics/final_protocol (baseline comparability), pairs
  regenerated under the new dispatch rule.
- dtu: built fresh (scene list = DA3 dtu baseline keys, eval = all frames;
  DTU is pose-only — no fscore downstream).

Output: workspace/ndispatch/<ds>/scene_manifest.json (VGGT train_arms layout).
DA3 runs sample on-the-fly with the same seeded function, so DA3/VGGT pairs
are identical by construction.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
import common as fg

FP = "artifacts/diagnostics/final_protocol"
OUT = "workspace/ndispatch"

DA3_DTU_SCENES = None  # filled from baselines.json


def scene_list_for(ds):
    src = json.load(open(f"{FP}/{ds}/scene_manifest.json"))
    return {s: src["scenes"][s] for s in src["scenes"]}, src


def build(ds):
    fg.set_dataset(ds)
    if ds == "dtu":
        bl = json.load(open("workspace/protocol_v2/baselines.json"))["da3"]["dtu"]["baseline"]
        scenes = {k: None for k in bl if not k.startswith("_")}
        src_meta = {"note": "dtu fresh build (pose-only)"}
    else:
        scenes, src_meta = scene_list_for(ds)

    out = {"dataset": ds,
           "note": f"N-dispatch v2 manifest (2026-09-20): pairs regenerated with "
                   f"protocol_v1 auto rule (N>=64 -> 16:4 else 8:4); "
                   + (f"eval32 inherited from final_protocol/{ds}"
                      if ds != "dtu" else "eval = all frames (pose-only)"),
           "run_root": f"{OUT}/{ds}", "scenes": {}}
    for scene in scenes:
        sd = fg.get_scene_data(scene)
        files = list(sd.image_files)
        N = len(files)
        proto = P.build_scene_protocol(files, scene, dataset=ds, n_train=10,
                                       n_shared=4, teacher_N=None)
        tN = proto["teacher_N"]
        ev = (scenes[scene]["eval32_frames"] if ds != "dtu"
              else list(range(N)))
        used = sorted(set(ev) | {f for p in proto["train_pairs"] + proto["probe_pairs"]
                                 for f in p["teacher_frames"]})
        out["scenes"][scene] = {
            "num_frames_total": N,
            "num_frames_with_gt_depth": scenes[scene]["num_frames_with_gt_depth"] if ds != "dtu" else N,
            "eval32_frames": ev,
            "adaptation_pool_size": N - len(set(ev)),
            "train_pairs": proto["train_pairs"],
            "probe_pairs": proto["probe_pairs"],
            "teacher_N": tN,
            "tau": proto.get("tau"),
            "strategy": proto.get("strategy"),
        }
        print(f"{ds}/{scene}: N={N} tN={tN} strategy={proto.get('strategy')} "
              f"eval={len(ev)} pairs={len(proto['train_pairs'])}")
    os.makedirs(f"{OUT}/{ds}", exist_ok=True)
    with open(f"{OUT}/{ds}/scene_manifest.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {OUT}/{ds}/scene_manifest.json ({len(out['scenes'])} scenes)")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["7scenes", "eth3d", "hiroom", "scannetpp", "dtu"]):
        build(ds)
