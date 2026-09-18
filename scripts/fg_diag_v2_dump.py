#!/usr/bin/env python3
"""Dump EVERY number from fg_diag_v2_*.json into flat human-readable CSVs.

No aggregation, no rounding beyond repr — one row per measurement, with full
coordinates (dataset/scene/state/input/frames/condition/group) so any cell in
the report can be traced back to its raw value.
"""
import csv
import json
import os
import sys

ROOT = "/root/autodl-tmp/Free-Geometry"
COMPS = ["maskdistill", "rkd_sh_d", "rkd_sh_a", "couple", "rel_rot", "rel_tdir"]
PAIR_FLOAT_KEYS = ["rot_chordal", "tdir_1mc", "rot_geodesic_deg", "tdir_angle_deg",
                   "t_rel_norm_teacher", "t_rel_norm_student"]
VIEW_KEYS = ["orth_err", "det", "depth_finite_frac"]


def frames_str(meta, iname):
    f = meta["inputs"].get(iname, {}).get("frames", [])
    return "+".join(str(x) for x in f)


def dump(dataset, path, outdir):
    d = json.load(open(path))
    os.makedirs(outdir, exist_ok=True)

    comp_rows, pair_rows, norm_rows, cos_rows, cc_rows = [], [], [], [], []

    for scene, sc in d.items():
        meta = sc["meta"]
        fr = {n: frames_str(meta, n) for n in meta["inputs"]}
        for state, st in sc["states"].items():
            # ---- M1 components ----
            for iname, conds in st["inputs"].items():
                for cond, payload in conds.items():
                    row = {"dataset": dataset, "scene": scene, "state": state,
                           "input": iname, "frames": fr[iname], "cond": cond}
                    row.update({k: repr(payload["components"][k]) for k in
                                payload["components"]})
                    comp_rows.append(row)
                    # ---- M1 per-camera-pair ----
                    for p in payload["pairs"]:
                        pr = {"dataset": dataset, "scene": scene, "state": state,
                              "input": iname, "frames": fr[iname], "cond": cond,
                              "cam_pair": p["ij"]}
                        for k in PAIR_FLOAT_KEYS:
                            if k in p:
                                pr[k] = repr(p[k])
                        for k in sorted(p):
                            if any(k.endswith(s) for s in
                                   [f"view{i}_{s}" for i in range(4) for s in VIEW_KEYS]):
                                pr[k] = repr(p[k])
                        pair_rows.append(pr)
            # ---- M2 gradient norms ----
            for key, g in st["grads"].items():
                probe, cond = key.split("/")
                for grp, norms in g["norms"].items():
                    norm_rows.append({
                        "dataset": dataset, "scene": scene, "state": state,
                        "probe": probe, "cond": cond, "group": grp,
                        **{k: repr(v) for k, v in norms.items()}})
                # ---- M2 cosines (incl. full 6x6) ----
                for key2, v in g["cos"].items():
                    grp, ckey = key2.split("|", 1)
                    cos_rows.append({
                        "dataset": dataset, "scene": scene, "state": state,
                        "probe": probe, "cond": cond, "group": grp,
                        "cosine": ckey, "value": repr(v)})
                # measured losses inside the gradient session (same forward)
                for c, v in g["losses"].items():
                    cos_rows.append({
                        "dataset": dataset, "scene": scene, "state": state,
                        "probe": probe, "cond": cond, "group": "-",
                        "cosine": f"loss[{c}]", "value": repr(v)})
        # ---- M3 check C ----
        for entry in sc.get("check_c", []):
            for br, b in entry["branches"].items():
                for where in ("after_on_input", "after_on_probe0_clean"):
                    if where not in b:
                        continue
                    for c, v in b[where].items():
                        base = (entry.get("before", {}).get(c)
                                if where == "after_on_input"
                                else (entry.get("before_probe0_clean") or {}).get(c))
                        cc_rows.append({
                            "dataset": dataset, "scene": scene,
                            "state": entry["state"], "update_input": entry["input"],
                            "branch": br, "eval_where": where, "component": c,
                            "before": repr(base) if base is not None else "",
                            "after": repr(v),
                            "delta": repr(v - base) if base is not None else "",
                            "grad_norm_preclip": repr(b["grad_norm_preclip"])})

    def w(name, rows):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r})
        p = os.path.join(outdir, name)
        with open(p, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(f"{p}: {len(rows)} rows, {len(keys)} cols")

    w(f"{dataset}_components.csv", comp_rows)
    w(f"{dataset}_camera_pairs.csv", pair_rows)
    w(f"{dataset}_grad_norms.csv", norm_rows)
    w(f"{dataset}_grad_cosines.csv", cos_rows)
    w(f"{dataset}_check_c.csv", cc_rows)

    # meta dump (ckpt hashes, param counts, frame lists)
    meta_rows = []
    for scene, sc in d.items():
        m = sc["meta"]
        meta_rows.append({
            "dataset": dataset, "scene": scene,
            "n_trainable_params": m["n_trainable_params"],
            "param_groups": json.dumps(m["param_groups"]),
            "param_group_numels": json.dumps(m["param_group_numels"]),
            "theta_base_ckpt": m["states"].get("theta_base", {}).get("ckpt", "seeded-reset"),
            "theta_base_sha256": m["states"].get("theta_base", {}).get("sha256", ""),
            "theta_rel_ckpt": m["states"].get("theta_rel", {}).get("ckpt", ""),
            "theta_rel_sha256": m["states"].get("theta_rel", {}).get("sha256", ""),
            "inputs_frames": json.dumps(m["inputs"]),
        })
    w(f"{dataset}_meta.csv", meta_rows)


if __name__ == "__main__":
    dump("scannetpp", f"{ROOT}/workspace/fg_diag_v2_scannetpp.json",
         f"{ROOT}/workspace/fg_diag_v2_raw")
    dump("7scenes", f"{ROOT}/workspace/fg_diag_v2_7scenes.json",
         f"{ROOT}/workspace/fg_diag_v2_raw")
