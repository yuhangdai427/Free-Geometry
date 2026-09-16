#!/usr/bin/env python3
"""Fusion + final analysis for ceiling_cov: merges patch_interp_f1's existing
a0/a100 rows with the new c2/c2p/c3r/t16/t32 variants; computes coverage of
the substitution ceiling by the trained losses and the longer-context ceiling
trend. Run AFTER ceiling_cov.py (GPU) in a fresh process (no CUDA)."""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import load_manifest  # noqa: E402
import patch_interp_f1 as P  # noqa: E402

OUT = P.OUT
NEW_VARIANTS = ["c2", "c2p", "c3r", "t16", "t32"]


def fusion_phase(scenes, max_pairs, workers, variants=None, out_name="ceiling_fusion_rows.jsonl"):
    from multiprocessing import Pool
    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    variants = variants or NEW_VARIANTS
    tasks = []
    for scene in scenes:
        n = len(manifest["scenes"][scene]["train_pairs"] + manifest["scenes"][scene]["probe_pairs"])
        if max_pairs:
            n = min(n, max_pairs)
        for pi in range(n):
            for name in variants:
                if os.path.exists(os.path.join(OUT, "depths", f"{scene}__p{pi:02d}__{name}.npz")):
                    tasks.append((scene, pi, name))
    print(f"[fusion] {len(tasks)} tasks on {workers} workers", flush=True)
    t0 = time.time()
    with open(os.path.join(OUT, out_name), "w") as f:
        with Pool(workers, initializer=P._fusion_init) as pool:
            for i, r in enumerate(pool.imap_unordered(P._fusion_one, tasks)):
                f.write(json.dumps(r) + "\n")
                f.flush()
                if (i + 1) % 30 == 0:
                    print(f"[fusion] {i + 1}/{len(tasks)} in {time.time() - t0:.0f}s", flush=True)
    import shutil
    shutil.rmtree(os.path.join(OUT, "fusion_work"), ignore_errors=True)


def main():
    import argparse
    import collections
    import statistics as st
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--skip_fusion", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest("artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json")
    scenes = sorted(manifest["scenes"])
    if not args.skip_fusion:
        fusion_phase(scenes, args.max_pairs, args.workers)

    # merge depth rows
    rows = {}
    for path in ("depth_rows.jsonl", "ceiling_rows.jsonl"):
        p = os.path.join(OUT, path)
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                rows[(r["scene"], r["pair"], r["variant"])] = r
    for path in ("fusion_rows.jsonl", "ceiling_fusion_rows.jsonl"):
        p = os.path.join(OUT, path)
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                k = (r["scene"], r["pair"], r["variant"])
                if k in rows:
                    rows[k].update({m: r[m] for m in ("acc", "comp", "overall", "precision", "recall", "fscore")})

    order = ["a0", "c2", "c2p", "c3r", "a100", "t16", "t32", "a100sham"]
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r["variant"]].append(r)

    def agg(rs, k):
        v = [r[k] for r in rs if k in r and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")

    def delta(name, key, base="a0"):
        b = {(r["scene"], r["pair"]): r for r in by[base]}
        out = []
        for r in by[name]:
            bb = b.get((r["scene"], r["pair"]))
            if bb and np.isfinite(r.get(key, np.nan)) and np.isfinite(bb.get(key, np.nan)):
                out.append((r[key] - bb[key], r["scene"], r["pair"]))
        return out

    lines = ["# coverage of the patch-substitution ceiling + longer-context ceiling", "",
             "72 pairs (6 scenes x 12), protocol of patch_interp_f1 (GT poses, pair-wide scale, repo TSDF).", "",
             "| variant | AbsRel | d1.25 | F1 | acc | comp |",
             "|---|---|---|---|---|---|"]
    for name in order:
        if name in by:
            rs = by[name]
            lines.append(f"| {name} | {agg(rs, 'absrel'):.4f} | {agg(rs, 'd125'):.4f} | "
                         f"{agg(rs, 'fscore'):.4f} | {agg(rs, 'acc'):.4f} | {agg(rs, 'comp'):.4f} |")

    lines += ["", "## paired deltas vs a0", "",
              "| variant | dAbsRel | wins | dF1 | wins | dAcc | wins |",
              "|---|---|---|---|---|---|---|"]
    for name in order[1:]:
        if name not in by:
            continue
        da, df, dc = delta(name, "absrel"), delta(name, "fscore"), delta(name, "acc")
        wa = sum(1 for x, _, _ in da if x < 0)
        wf = sum(1 for x, _, _ in df if x > 0)
        wc = sum(1 for x, _, _ in dc if x < 0)
        lines.append(f"| {name} | {st.mean([x for x, _, _ in da]):+.5f} | {wa}/{len(da)} | "
                     f"{st.mean([x for x, _, _ in df]):+.4f} | {wf}/{len(df)} | "
                     f"{st.mean([x for x, _, _ in dc]):+.4f} | {wc}/{len(dc)} |")

    # coverage: trained gain / substitution ceiling, on ceiling-positive pairs
    lines += ["", "## coverage of the ceiling (ceiling = a100)", ""]
    for key, better in (("absrel", "neg"), ("fscore", "pos"), ("acc", "neg")):
        ceil = { (s, p): x for x, s, p in delta("a100", key) }
        lines.append(f"metric {key}:")
        for name in ("c2", "c2p", "c3r", "t16", "t32"):
            if name not in by:
                continue
            tr = { (s, p): x for x, s, p in delta(name, key) }
            common = [k for k in tr if k in ceil]
            pos_pairs = [k for k in common if (ceil[k] < 0 if better == "neg" else ceil[k] > 0)]
            for tag, ks in (("all", common), ("ceiling>0", pos_pairs)):
                if not ks:
                    continue
                mc = np.mean([ceil[k] for k in ks])
                mt = np.mean([tr[k] for k in ks])
                cov = mt / mc if mc != 0 else float("nan")
                lines.append(f"  {name:4s} [{tag:10s} n={len(ks):2d}] gain={mt:+.5f} ceiling={mc:+.5f} coverage={cov * 100:+.1f}%")

    # per-scene F1 for the main variants
    main_v = ["a0", "c2", "c3r", "a100", "t16", "t32"]
    lines += ["", "## per-scene F1", "", "| scene | " + " | ".join(main_v) + " |", "|---|---|---|---|---|---|---|"]
    for scene in scenes:
        vals = []
        for name in main_v:
            rs = [r for r in by.get(name, []) if r["scene"] == scene]
            vals.append(f"{agg(rs, 'fscore'):.4f}")
        lines.append(f"| {scene} | " + " | ".join(vals) + " |")

    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "coverage_summary.md"), "w") as f:
        f.write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    main()
