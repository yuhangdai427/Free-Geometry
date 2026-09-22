#!/usr/bin/env python3
"""Build the dashboard data blob: per-scene per-model per-steps gradient
trajectories (4 components per update+pair) + metrics vs baseline.
Output: /tmp/dash_data.json (consumed by the HTML template injector)."""
import csv, glob, json, re, statistics as st

BASE = json.load(open("workspace/protocol_v2/baselines.json"))
DATASETS = ["eth3d", "7scenes", "scannetpp", "hiroom"]

def da3_steps(ds, scene_dir, u):
    """Merge grad_components.csv (huber/cos/gn) + grad_rel.csv (rot/tdir)."""
    gc_p, gr_p = f"{scene_dir}/grad_components.csv", f"{scene_dir}/grad_rel.csv"
    try:
        gc = {int(r["update"]): r for r in csv.DictReader(open(gc_p))}
        gr = {int(r["update"]): r for r in csv.DictReader(open(gr_p))}
    except Exception:
        return None
    rows = []
    for u_ in sorted(gc):
        if u_ not in gr:
            continue
        a, b = gc[u_], gr[u_]
        rows.append([u_, int(a["pair"]),
                     round(float(a["g_mse"]), 5), round(float(a["g_cos"]), 5),
                     round(float(b["g_rot"]), 5), round(float(b["g_tdir"]), 5),
                     round(float(b["l_rot"]), 5), round(float(a["gn"]), 4)])
    return rows or None

def vggt_steps(ds, u):
    """Parse gcomp lines: [step, pair, g_huber, g_cos, g_rot, g_tdir, l_rot, gn]."""
    out = {}
    try:
        txt = open(f"logs/ov_vggt_{ds}_u{u}.log", errors="ignore").read()
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.match(r"\[(.+?)\] gcomp step (\d+) pair=(\d+) L_huber=([\d.]+) L_cos=[\d.]+ "
                     r"g_huber=([\d.e-]+) g_cos=([\d.e-]+) cos_share=[\d.]+ dir=[\d.-]+ "
                     r"gn=([\d.a-z]+) lr=[\d.e+-]+ \| rot: L=([\d.]+) g=([\d.]+) "
                     r"\| tdir: L=[\d.]+ g=([\d.]+)", line)
        if m:
            sc = m.group(1); k = int(m.group(2)); p = int(m.group(3))
            gn = m.group(7)
            gn = float(gn) if gn != "nan" else None
            out.setdefault(sc, []).append([k, p,
                round(float(m.group(5)), 5), round(float(m.group(6)), 5),
                round(float(m.group(9)), 5), round(float(m.group(10)), 5),
                round(float(m.group(8)), 5), gn])
    return out

def da3_metrics(ds):
    """{(scene, u): (auc, f1, absrel)} from per-cell logs (scene parsed from line)."""
    out = {}
    for f in glob.glob(f"logs/ov_da3_{ds}_*_u*.log"):
        mu = re.search(r"_u(\d+)\.log$", f)
        m = re.search(r"\[(.+?)\] huber/vggt/enc u\d+k1 \(cam_dec\): AUC=([\d.]+) F1=([\d.]+) abs_rel=([\d.]+)",
                      open(f, errors="ignore").read())
        if m and mu:
            out[(m.group(1), int(mu.group(1)))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out

def vggt_metrics(ds):
    out = {}
    for u in [100, 50]:
        try:
            txt = open(f"logs/ov_vggt_eval_{ds}_u{u}.log", errors="ignore").read()
        except Exception:
            continue
        for m in re.finditer(r"\[(.+?)\] TTA: AUC@3=([\d.]+) F1=([\d.]+) abs_rel=([\d.]+)", txt):
            out[(m.group(1), u)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out

data = {"datasets": DATASETS, "scenes": {}}
for ds in DATASETS:
    dmet, vmet = da3_metrics(ds), vggt_metrics(ds)
    vsteps = {u: vggt_steps(ds, u) for u in [100, 50]}
    b_da3 = BASE["da3"].get(ds, {}).get("baseline", {})
    b_vg = BASE["vggt"].get(ds, {}).get("baseline", {})
    scenes = set()
    for (s, u) in list(dmet) + list(vmet):
        scenes.add(s)
    for s in sorted(scenes):
        entry = {"metrics": {}}
        bd = b_da3.get(s, {}); bv = b_vg.get(s, {})
        for model, met, bs in [("da3", dmet, bd), ("vggt", vmet, bv)]:
            for u in [100, 50]:
                cell = {}
                if (s, u) in met:
                    a, f, ar = met[(s, u)]
                    cell.update(auc=a, f1=f, abs_rel=ar)
                    if "auc03" in bs:
                        cell.update(base_auc=bs["auc03"], base_f1=bs["fscore"])
                if cell:
                    entry["metrics"][f"{model}_u{u}"] = cell
        # gradients
        for u in [100, 50]:
            rows = da3_steps(ds, f"workspace/overnight/da3_{ds}_u{u}/{s}", u)
            if rows:
                entry.setdefault("grads", {})[f"da3_u{u}"] = rows
            vr = vsteps[u].get(s)
            if vr:
                entry.setdefault("grads", {})[f"vggt_u{u}"] = vr
        if entry.get("metrics") or entry.get("grads"):
            data["scenes"].setdefault(ds, {})[s] = entry

n_sc = sum(len(v) for v in data["scenes"].values())
n_cells = sum(len(e.get("grads", {})) for ds in data["scenes"].values() for e in ds.values() if isinstance(ds, dict)) if False else \
          sum(len(e.get("grads", {})) for dsv in data["scenes"].values() for e in dsv.values())
json.dump(data, open("/tmp/dash_data.json", "w"))
import os
print(f"scenes={n_sc} grad_cells={n_cells} size={os.path.getsize('/tmp/dash_data.json')/1e6:.2f}MB")
for ds in DATASETS:
    print(f"  {ds}: {len(data['scenes'].get(ds, {}))} scenes")
