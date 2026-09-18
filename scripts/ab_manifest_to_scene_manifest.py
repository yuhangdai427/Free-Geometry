#!/usr/bin/env python3
"""Merge per-scene v2ab AB manifests into VGGT train_arms scene_manifest format.

Source A: workspace/protocol_v2/ab_manifests/<ds>/<scene>.json (v2ab per-scene
protocols built by scripts/build_ab_context_manifests.py; the file's internal
"scene" field is authoritative, so hiroom's '/'-containing names survive).
Source B: artifacts/diagnostics/final_protocol/<ds>/scene_manifest.json — the
VGGT baseline manifest. Fields train_arms does not derive from the pairs are
INHERITED per scene (file_sha256, num_frames_with_gt_depth,
adaptation_pool_size, seed_scene_select); eval32_frames is copied VERBATIM
from the reference so eval frames stay bit-identical to the baselines.
train_pairs / probe_pairs come from the v2ab manifests (10 A/B dual-context
train tasks at teacher_N=8 with teacher_frames_B; 2 single-context probes).

Datasets without a final_protocol reference (dtu, dtu64): eval32_frames falls
back to the v2ab manifest's eval_frames — same convention as
build_final_manifest (N>=100 -> random.Random(42) benchmark-100; else all) —
num_frames_with_gt_depth is computed from the dataset aux, and the fallback is
flagged in the note and in the per-scene eval_source check table.

 train_arms --v2_ab contract validated here (mirrors train_arms.py:2839-2862):
  - student_frames == teacher_frames[[0,2,4,6]] for every train/probe pair
  - every train pair carries teacher_frames_B of equal length, shared at the
    same even slots; probe pairs stay single-context
  - train shared-group keys and A-union-B keys are unique; probe keys are
    disjoint from train keys
  - eval32_frames identical to the final_protocol reference (ref datasets)

Usage:
  python scripts/ab_manifest_to_scene_manifest.py \
      --datasets eth3d 7scenes scannetpp hiroom dtu dtu64 \
      --ab_root workspace/protocol_v2/ab_manifests \
      --ref_root artifacts/diagnostics/final_protocol \
      --out workspace/protocol_v2/ab_scene_manifests
  # re-validate already-written manifests:
  python scripts/ab_manifest_to_scene_manifest.py --check_only ...
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "free_geometry"))

STUDENT_SLOTS = [0, 2, 4, 6]
REF_DATASETS = ("eth3d", "7scenes", "scannetpp", "hiroom")


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# validation (shared by merge and --check_only)
# ---------------------------------------------------------------------------
def check_scene_entry(scene: str, sc: Dict, require_ab: bool = True) -> Tuple[List[str], List[str]]:
    """train_arms --v2_ab contract for one manifest scene entry. Returns
    (issues, warnings); duplicated train keys on scenes the v2ab builder marked
    degraded (tiny-N fallback fill) are warnings, not issues."""
    issues: List[str] = []
    warnings: List[str] = []
    degraded = bool(sc.get("degraded", False))
    train, probe = sc["train_pairs"], sc["probe_pairs"]
    shared_keys, union_keys = set(), set()
    for pi, p in enumerate(train):
        t = p["teacher_frames"]
        if p["student_frames"] != [t[i] for i in STUDENT_SLOTS]:
            issues.append(f"{scene}: train{pi} student != teacher[slots]")
        if len(set(t)) != len(t):
            issues.append(f"{scene}: train{pi} teacher A has duplicate frames")
        ks, ku = tuple(p["student_frames"]), tuple(sorted(set(t) | set(p.get("teacher_frames_B", []))))
        dup = []
        if ks in shared_keys:
            dup.append("shared group duplicated")
        if ku in union_keys:
            dup.append("A-union-B duplicated")
        if dup:
            (warnings if degraded else issues).append(f"{scene}: train{pi} {', '.join(dup)}")
        shared_keys.add(ks)
        union_keys.add(ku)
        if require_ab:
            tb = p.get("teacher_frames_B")
            if tb is None:
                issues.append(f"{scene}: train{pi} missing teacher_frames_B")
            else:
                if len(tb) != len(t):
                    issues.append(f"{scene}: train{pi} teacher_frames_B length mismatch")
                if any(tb[i] != t[i] for i in STUDENT_SLOTS):
                    issues.append(f"{scene}: train{pi} B slots != shared")
                if len(set(tb)) != len(tb):
                    issues.append(f"{scene}: train{pi} teacher B has duplicate frames")
    probe_collide = union_keys | {tuple(sorted(x["teacher_frames"])) for x in train}
    for pi, p in enumerate(probe):
        t = p["teacher_frames"]
        if p["student_frames"] != [t[i] for i in STUDENT_SLOTS]:
            issues.append(f"{scene}: probe{pi} student != teacher[slots]")
        if tuple(p["student_frames"]) in shared_keys:
            (warnings if degraded else issues).append(
                f"{scene}: probe{pi} shared key collides with train")
        if tuple(sorted(t)) in probe_collide:
            (warnings if degraded else issues).append(
                f"{scene}: probe{pi} teacher list collides with a train union")
    ev = set(sc.get("eval32_frames", []))
    if not ev:
        issues.append(f"{scene}: empty eval32_frames")
    return issues, warnings


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def n_gt_frames(ds: str, scene: str) -> Optional[int]:
    """num_frames_with_gt_depth computed like build_final_manifest (no-ref datasets)."""
    try:
        import common as fg_common
        fg_common.set_dataset(ds)
        data = fg_common.get_scene_data(scene)
        return int(sum(1 for p in data.aux.gt_depth_files if os.path.exists(p)))
    except Exception as e:  # informational field: never fail the merge on it
        print(f"  [{ds}/{scene}] WARN num_frames_with_gt_depth fallback ({e!r})")
        return None


def merge_dataset(ds: str, ab_dir: str, ref_path: Optional[str], out_dir: str,
                  ab_root: str) -> Dict:
    ref = load_json(ref_path) if ref_path else None
    rows = []
    scenes: Dict = {}
    for path in sorted(glob.glob(os.path.join(ab_dir, "*.json"))):
        if path.endswith(".bak"):
            continue
        ab = load_json(path)
        scene = ab["scene"]  # authoritative (hiroom names contain '/')
        ref_sc = ref["scenes"].get(scene) if ref else None
        has_ref = ref is not None
        if has_ref and ref_sc is None:
            print(f"  [{ds}/{scene}] WARN not in final_protocol reference; v2ab fallback fields")
        eval32 = list(ref_sc["eval32_frames"]) if ref_sc else list(ab["eval_frames"])
        eval_src = "final_protocol" if ref_sc else "v2ab_convention"
        oob = [i for i in eval32 if i >= ab["N"]]
        n_gt = ref_sc["num_frames_with_gt_depth"] if ref_sc else n_gt_frames(ds, scene)
        if n_gt is None:
            n_gt = ab["N_valid"]
        entry = {
            "num_frames_total": ab["N"],
            "file_sha256": (ref_sc or {}).get("file_sha256", {}),
            "num_frames_with_gt_depth": n_gt,
            "adaptation_pool_size": (ref_sc or {}).get("adaptation_pool_size", ab["N"]),
            "teacher_N": ab["teacher_N"],
            "tau": ab["tau"],
            "strategy": ab["strategy"],
            "degraded": ab["degraded"],
            "degrade_reasons": ab["degrade_reasons"],
            "eval32_frames": eval32,
            "train_pairs": ab["train_pairs"],
            "probe_pairs": ab["probe_pairs"],
        }
        scenes[scene] = entry
        rows.append({"scene": scene, "N": ab["N"], "n_train": len(ab["train_pairs"]),
                     "n_probe": len(ab["probe_pairs"]), "eval_n": len(eval32),
                     "eval_oob": len(oob), "degraded": ab["degraded"],
                     "ab_overlap_mean": ab["ab_overlap_mean"], "eval_src": eval_src})

    note = (
        f"protocol v2 phase-C A/B context manifest (merged {np.datetime64('now')}): "
        f"train_pairs/probe_pairs from {ab_root}/{ds} (v2ab: 10 tasks, one shared "
        f"group S(4) under two 8-frame teacher contexts A/B, shared at slots "
        f"[0,2,4,6], extras SIFT-ranked disjoint-preferred; 2 single-context "
        f"probes; per-pair teacher_frames_B / ab_overlap / n_candidates kept). "
    )
    if ref is not None:
        note += ("eval32_frames / file_sha256 / num_frames_with_gt_depth / "
                 "adaptation_pool_size inherited VERBATIM from "
                 f"final_protocol/{ds} (baseline comparability). ")
    else:
        note += ("no final_protocol reference for this dataset: eval32_frames "
                 "follows the v2ab/build_final_manifest convention "
                 "(N>=100 -> random.Random(42) benchmark-100, else all frames). ")
    note += "Consumes with train_arms.py --v2_ab."

    manifest = {"dataset": ds,
                "run_root": os.path.join(ROOT, out_dir, ds),
                "seed_scene_select": (ref or {}).get("seed_scene_select", 43),
                "note": note,
                "scenes": scenes}
    os.makedirs(os.path.join(ROOT, out_dir, ds), exist_ok=True)
    out_path = os.path.join(ROOT, out_dir, ds, "scene_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f)
    return {"path": out_path, "has_ref": ref is not None, "rows": rows,
            "scenes": scenes, "dataset": manifest}


def print_table(ds: str, rows: List[Dict]) -> None:
    print(f"\n== {ds} ({len(rows)} scenes) ==", flush=True)
    print(f"{'scene':<38} {'N':>5} {'trn':>4} {'prb':>4} {'eval':>5} {'oob':>4} "
          f"{'degr':>5} {'ab_ov':>6}  eval_src")
    for r in rows:
        name = r["scene"] if len(r["scene"]) <= 37 else "..." + r["scene"][-34:]
        ov = f"{r['ab_overlap_mean']:.2f}" if r["ab_overlap_mean"] is not None else "-"
        print(f"{name:<38} {r['N']:>5} {r['n_train']:>4} {r['n_probe']:>4} "
              f"{r['eval_n']:>5} {r['eval_oob']:>4} {str(r['degraded']):>5} {ov:>6}  "
              f"{r['eval_src']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+",
                    default=["eth3d", "7scenes", "scannetpp", "hiroom", "dtu", "dtu64"])
    ap.add_argument("--ab_root", default=os.path.join(ROOT, "workspace", "protocol_v2", "ab_manifests"))
    ap.add_argument("--ref_root", default=os.path.join(ROOT, "artifacts", "diagnostics", "final_protocol"))
    ap.add_argument("--out", default=os.path.join(ROOT, "workspace", "protocol_v2", "ab_scene_manifests"))
    ap.add_argument("--check_only", action="store_true",
                    help="skip writing; validate existing merged manifests and re-check "
                         "eval32 identity against the reference")
    args = ap.parse_args()

    all_issues: List[str] = []
    for ds in args.datasets:
        ab_dir = os.path.join(ROOT, args.ab_root, ds)
        ref_path = os.path.join(ROOT, args.ref_root, ds, "scene_manifest.json")
        has_ref_file = os.path.exists(ref_path)
        if not has_ref_file:
            ref_path = None
        out_path = os.path.join(ROOT, args.out, ds, "scene_manifest.json")

        if args.check_only:
            if not os.path.exists(out_path):
                print(f"[{ds}] merged manifest missing: {out_path}")
                all_issues.append(f"{ds}: missing merged manifest")
                continue
            manifest = load_json(out_path)
            ref = load_json(ref_path) if ref_path else None
            rows = []
            scenes = manifest["scenes"]
        else:
            if not os.path.isdir(ab_dir):
                print(f"[{ds}] no ab manifests at {ab_dir}; skipped")
                all_issues.append(f"{ds}: no ab manifest dir")
                continue
            res = merge_dataset(ds, ab_dir, ref_path, args.out, args.ab_root)
            manifest = res["dataset"]
            scenes = res["scenes"]
            rows = res["rows"]
            ref = load_json(ref_path) if ref_path else None
            print(f"[{ds}] wrote {res['path']} (ref={'yes' if has_ref_file else 'NO -> v2ab convention'})")

        # validation pass (both modes)
        ds_issues = []
        ds_warns = []
        for scene, sc in scenes.items():
            iss, wrn = check_scene_entry(scene, sc, require_ab=True)
            ds_issues += iss
            ds_warns += wrn
            if ref is not None and scene in ref["scenes"]:
                if sc["eval32_frames"] != list(ref["scenes"][scene]["eval32_frames"]):
                    ds_issues.append(f"{scene}: eval32_frames != final_protocol reference")
            oob = [i for i in sc["eval32_frames"] if i >= sc["num_frames_total"]]
            if oob:
                ds_issues.append(f"{scene}: {len(oob)} eval32 frame(s) out of bounds "
                                 f"(>= num_frames_total={sc['num_frames_total']})")
        if not args.check_only:
            print_table(ds, rows)
        for w in ds_warns:
            print(f"  [{ds}] WARN(degraded scene): {w}", flush=True)
        if ds_issues:
            all_issues += [f"[{ds}] {i}" for i in ds_issues]
            for i in ds_issues:
                print(f"  [{ds}] ISSUE: {i}", flush=True)
        else:
            print(f"  [{ds}] checks OK: slots, A/B contract, dedup, probe disjointness, "
                  f"eval32 identity ({len(ds_warns)} degraded-scene warnings)", flush=True)

    print("\n==== SELF-CHECK SUMMARY ====")
    if all_issues:
        print(f"{len(all_issues)} issue(s):")
        for i in all_issues:
            print(f"  - {i}")
        sys.exit(1)
    print("all manifests OK")


if __name__ == "__main__":
    main()
