"""Build a two_stage (combo: fixed-slot + endpoint-anchored) manifest for VGGT
scannetpp, matching the DA3 winner's sampling (rkdc1h + two_stage 0.7).

Pairs: 10 fixed-slot 16:4 + 10 endpoint-anchored (first+last+2 mids, L=4
forced — VGGT's STUDENT_INDICES assumes 4-frame students), kind-tagged,
seeded deterministically per scene. Output: a scene_manifest.json clone of the
canonical scannetpp manifest with train_pairs replaced (probe/eval frames
untouched).
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))

import common as fg_common
from depth_anything_3.test_time_adaption import protocol_v1 as P

SRC = "artifacts/diagnostics/final_protocol/scannetpp/scene_manifest.json"
DST = "artifacts/diagnostics/final_protocol/scannetpp_2stage/scene_manifest.json"


def se_pair_l4(N, tn, dense, sc, rng):
    """Endpoint-anchored pair with exactly 4 student frames."""
    for _ in range(50):
        teacher, shared = P.build_pair(N, tn, dense, sc, rng, n_shared=4,
                                       selfevo=True, selfevo_lmin=4)
        if len(shared) == 4:
            return teacher, shared
    return teacher, shared  # give up: keep whatever we have


def main():
    fg_common.set_dataset("scannetpp")
    manifest = json.load(open(SRC))
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    for scene, sc_entry in manifest["scenes"].items():
        N = sc_entry["num_frames_total"]
        image_files = list(fg_common.get_scene_data(scene).image_files)
        assert len(image_files) == N, f"{scene}: {len(image_files)} != {N}"
        tau = P.compute_tau(image_files)
        dense = tau > P.TAU_THRESHOLD
        sift = P.SiftCache(image_files) if dense else None
        rng = random.Random(P.stable_seed("2stage", "scannetpp", scene))
        pairs, seen_t, seen_s = [], set(), set()
        while len([p for p in pairs if p["kind"] == "fixed"]) < 10:
            teacher, shared = P.build_pair(N, 16, dense, sift, rng, n_shared=4)
            kt, ks = tuple(sorted(teacher)), tuple(shared)
            if kt in seen_t or ks in seen_s:
                continue
            seen_t.add(kt); seen_s.add(ks)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared),
                          "teacher_N": 16, "n_shared": 4, "kind": "fixed"})
        while len([p for p in pairs if p["kind"] == "se"]) < 10:
            teacher, shared = se_pair_l4(N, 16, dense, sift, rng)
            kt, ks = tuple(sorted(teacher)), tuple(shared)
            if kt in seen_t or ks in seen_s:
                continue
            seen_t.add(kt); seen_s.add(ks)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared),
                          "teacher_N": 16, "n_shared": 4, "kind": "se"})
        sc_entry["train_pairs"] = pairs
        print(f"[{scene}] N={N} tau={tau:.3f} dense={dense} "
              f"fixed={sum(p['kind']=='fixed' for p in pairs)} se={sum(p['kind']=='se' for p in pairs)}",
              flush=True)
    json.dump(manifest, open(DST, "w"), indent=1)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
