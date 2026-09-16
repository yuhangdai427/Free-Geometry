#!/usr/bin/env python3
"""Extend each dev-scene train pair's 8 teacher frames to 16/32-view contexts.

For every scene x train pair in the bakeoff manifest, extra frames are drawn
from the scene's non-eval pool (frames with GT depth minus eval32_frames),
nearest-first by frame-number distance to the pair's original 8 frames
(tie-break: smaller frame index). The original 8 frames keep slots 0..7 and
the shared 4 frames stay at slots [0,2,4,6]; extras are appended. Eval-frame
disjointness is asserted, mirroring common.verify_disjoint.

Output: artifacts/diagnostics/context_scaling/manifest.json
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import STUDENT_INDICES, frames_with_gt_depth, load_manifest

CTX_SIZES = (16, 32)
OUT_DEFAULT = "artifacts/diagnostics/context_scaling/manifest.json"


def pick_extras(base8, pool, n_extra):
    base = set(base8)
    cands = [f for f in pool if f not in base]
    cands.sort(key=lambda f: (min(abs(f - b) for b in base8), f))
    return cands[:n_extra]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_manifest", default="artifacts/diagnostics/bakeoff_v1/scene_manifest.json")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    man = load_manifest(args.scene_manifest)
    out = {"source_manifest": args.scene_manifest, "ctx_sizes": CTX_SIZES,
           "scenes": {}}
    for scene in sorted(man["scenes"]):
        sc = man["scenes"][scene]
        eval_set = set(sc["eval32_frames"])
        ok, _ = frames_with_gt_depth(scene)
        pool = sorted(f for f in ok if f not in eval_set)
        assert len(pool) == sc["adaptation_pool_size"], (
            f"{scene}: pool {len(pool)} != manifest {sc['adaptation_pool_size']}")

        pairs = []
        for pi, pair in enumerate(sc["train_pairs"]):
            base8 = list(pair["teacher_frames"])
            shared4 = list(pair["student_frames"])
            assert shared4 == [base8[i] for i in STUDENT_INDICES]
            assert not (eval_set & set(base8)), f"{scene} pair{pi}: eval leak in base8"
            ctx = {"8": {"frames": base8, "shared_slots": list(STUDENT_INDICES)}}
            for n in CTX_SIZES:
                extras = pick_extras(base8, pool, n - 8)
                frames = base8 + extras
                assert len(set(frames)) == len(frames)
                assert not (eval_set & set(frames)), f"{scene} pair{pi}: eval leak ctx{n}"
                ctx[str(n)] = {
                    "frames": frames,
                    "shared_slots": list(STUDENT_INDICES),
                    "n_requested": n,
                    "n_actual": len(frames),
                }
            pairs.append({"pair_id": pi, "shared_frames": shared4, "ctx": ctx})
        out["scenes"][scene] = {"adaptation_pool_size": len(pool),
                                "eval32_frames": sc["eval32_frames"],
                                "train_pairs": pairs}
        print(f"[{scene}] pool={len(pool)} ctx32 extras head: "
              f"{pairs[0]['ctx']['32']['frames'][8:14]}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
