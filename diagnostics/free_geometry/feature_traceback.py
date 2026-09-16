#!/usr/bin/env python3
"""Stage 2 (feature traceback): attribute the long-context gain to patch-token
movement, and find GT-free signals that predict where ctx16 fixes ctx4.

Per scene (frozen VGGT, analysis only):
1. gain[p] = err_ctx4[p] - err_ctx16[p] per patch (>0 = long context rescued),
   err maps from Stage 1 (context_effect depths/<scene>.npz, perpatch_logres2
   scene-shared-scale residuals).
2. disp_l[p] = || to_norm(h_ctx16,l)[p] - to_norm(h_ctx4,l)[p] ||_2 for
   l in [4,11,17,23] (patch tokens only; features re-computed with the same
   frozen forwards as Stage 1 - cudnn-deterministic, so identical). Spearman
   corr(disp_l, gain) per layer per scene + pooled.
3. Right/wrong attribution with GT used ONLY for labels:
   right = err_ctx16[p] < err_ctx4[p] (strict; ties excluded - they are mostly
   fully GT-invalid patches). GT-free candidate signals: ctx16 depth_conf
   (per-patch mean), disp_l, post-norm feature norm ||to_norm(h_ctx16,l)||,
   cross-layer disp consistency (-CV of the 4 layer disps), and products
   conf_z x disp_z (z-scored per scene). ROC-AUC per signal per scene + pooled.
4. summary.md: which layer's disp tracks the gain, AUC table, best GT-free
   predictor, implications for knowledge transfer.

Outputs: per_scene/<scene>.npz (gain/disp/conf/labels), spearman.csv, auc.csv,
summary.md. No GT is used anywhere except the offline err maps from Stage 1.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import get_scene_data, load_image_model, load_manifest  # noqa: E402
import modeling as M  # noqa: E402

MANIFEST_DEFAULT = "artifacts/diagnostics/bakeoff_v2_transductive/scene_manifest.json"
STAGE1_DEFAULT = "artifacts/diagnostics/context_effect"
OUT_DIR_DEFAULT = "artifacts/diagnostics/feature_traceback"
PROGRESS_MD = os.path.join("artifacts", "diagnostics", "PROGRESS.md")
CTX_PAIR = (4, 16)

SPEAR_FIELDS = ["scene", "layer", "spearman", "n_patches"]
AUC_FIELDS = ["scene", "feature", "auc", "n_right", "n_wrong", "frac_excluded"]


def progress(msg):
    line = f"- [{time.strftime('%H:%M')}] {msg}\n"
    try:
        with open(PROGRESS_MD, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    print(f"[progress] {msg}", flush=True)


def roc_auc(score, label):
    """Mann-Whitney AUC with average ranks for ties. label: 1=right, 0=wrong."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, np.int64)
    n1 = int((label == 1).sum())
    n0 = int((label == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score, method="average")
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def zscore(x):
    x = np.asarray(x, np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


@torch.no_grad()
def cache_ctx_patch_feats(vggt, images, ctx):
    """Chunk forwards at ctx; returns {layer: [32,P,2048] fp16 CPU}."""
    n_chunks = images.shape[1] // ctx
    out = {l: [] for l in M.TAP_LAYERS}
    for k in range(n_chunks):
        images_c = images[:, k * ctx:(k + 1) * ctx].contiguous()
        feats24, psi = M.aggregator_all(vggt, images_c)
        for l in M.TAP_LAYERS:
            out[l].append(M.to_patch(feats24[l]).half().cpu())
        del feats24, images_c
        torch.cuda.empty_cache()
    return {l: torch.cat(v, dim=1) for l, v in out.items()}  # [1,32,P,C]


@torch.no_grad()
def norm_feats(depth_head, h_cpu, device="cuda"):
    """h_cpu [1,S,P,C] fp16 -> post-norm fp32 [S,P,C] on device, in view chunks."""
    S = h_cpu.shape[1]
    outs = []
    for i in range(0, S, 8):
        h = h_cpu[:, i:i + 8].float().to(device)
        outs.append(M.to_norm(depth_head, h).cpu())
        del h
    return torch.cat(outs, dim=1).squeeze(0)  # [S,P,C]


def patch_mean_conf(conf, patch_hw):
    """conf [S,H,W] -> per-patch mean [S,P]."""
    S, H, W = conf.shape
    ph, pw = patch_hw
    return conf.reshape(S, ph, H // ph, pw, W // pw).mean(axis=(2, 4)).reshape(S, ph * pw)


def run_scene(vggt, scene, scene_data, ev, stage1_dir, device="cuda"):
    z = np.load(os.path.join(stage1_dir, "depths", f"{scene}.npz"))
    patch_hw = tuple(int(v) for v in z["patch_hw"])
    e4, e16 = z["e_ctx4"].astype(np.float64), z["e_ctx16"].astype(np.float64)
    S = e4.shape[0]
    assert S == 32 and e16.shape == e4.shape
    conf16 = np.stack([z[f"c_ctx16_f{pos:03d}"].astype(np.float32) for pos in range(32)], 0)
    confp = patch_mean_conf(conf16, patch_hw)  # [32,P]

    imgs = [load_image_model(scene_data.image_files[i]) for i in ev]
    arr = np.stack(imgs, 0)
    images = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)

    feats = {}
    for ctx in CTX_PAIR:
        feats[ctx] = cache_ctx_patch_feats(vggt, images, ctx)
    depth_head = vggt.depth_head

    disp, fnorm16 = {}, {}
    for l in M.TAP_LAYERS:
        h4n = norm_feats(depth_head, feats[4][l], device)   # [32,P,C]
        h16n = norm_feats(depth_head, feats[16][l], device)
        disp[l] = (h16n - h4n).norm(dim=-1).numpy()         # [32,P]
        fnorm16[l] = h16n.norm(dim=-1).numpy()
        del h4n, h16n
    del feats, images
    torch.cuda.empty_cache()

    gain = e4 - e16
    res = {"gain": gain.astype(np.float32), "confp": confp.astype(np.float32),
           "right": (e16 < e4).astype(np.int8), "tie": (e16 == e4).astype(np.int8)}
    for l in M.TAP_LAYERS:
        res[f"disp{l}"] = disp[l].astype(np.float32)
        res[f"fnorm16_{l}"] = fnorm16[l].astype(np.float32)

    # --- Spearman(disp_l, gain) per layer
    spear_rows = []
    for l in M.TAP_LAYERS:
        rho = float(spearmanr(disp[l].ravel(), gain.ravel()).statistic)
        spear_rows.append({"scene": scene, "layer": l, "spearman": rho,
                           "n_patches": int(gain.size)})

    # --- GT-free signals for right/wrong
    disp_stack = np.stack([zscore(disp[l].ravel()) for l in M.TAP_LAYERS], 0)  # [4,N]
    disp_mean_z = disp_stack.mean(0)
    disp_raw = np.stack([disp[l].ravel() for l in M.TAP_LAYERS], 0)
    disp_cv = -(disp_raw.std(0) / (np.abs(disp_raw.mean(0)) + 1e-8))  # consistency: -CV
    fnorm_stack_z = np.stack([zscore(fnorm16[l].ravel()) for l in M.TAP_LAYERS], 0)
    conf_z = zscore(confp.ravel())
    signals = {
        "conf16": confp.ravel(),
        "disp4": disp[4].ravel(), "disp11": disp[11].ravel(),
        "disp17": disp[17].ravel(), "disp23": disp[23].ravel(),
        "disp_mean_z": disp_mean_z,
        "disp_consistency": disp_cv,
        "fnorm16_23": fnorm16[23].ravel(),
        "fnorm16_mean_z": fnorm_stack_z.mean(0),
        "conf_x_disp23": conf_z * zscore(disp[23].ravel()),
        "conf_x_disp_mean": conf_z * disp_mean_z,
    }
    label = res["right"].ravel()
    tie = res["tie"].ravel().astype(bool)
    keep = ~tie
    auc_rows = []
    for name, sig in signals.items():
        auc_rows.append({"scene": scene, "feature": name,
                         "auc": roc_auc(sig[keep], label[keep]),
                         "n_right": int(label[keep].sum()),
                         "n_wrong": int((1 - label)[keep].sum()),
                         "frac_excluded": float(tie.mean())})
    # top-decile gain: layer share of disp
    flat_gain = gain.ravel()
    thr = np.quantile(flat_gain, 0.9)
    top = flat_gain >= thr
    shares = {l: float(disp[l].ravel()[top].sum() /
                       max(sum(disp[m].ravel()[top].sum() for m in M.TAP_LAYERS), 1e-12))
              for l in M.TAP_LAYERS}
    res["top_decile_gain_disp_share"] = np.asarray([shares[l] for l in M.TAP_LAYERS],
                                                   dtype=np.float32)
    return res, spear_rows, auc_rows


def _fmt(x, nd=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_summary(args, scenes, all_spear, all_auc, scene_res):
    lines = ["# Feature traceback: patch-token attribution of the long-context gain",
             "",
             f"Stage-1 dir: `{args.stage1_dir}`; scenes: {', '.join(scenes)}. "
             "gain = err_ctx4 - err_ctx16 (per-patch centered squared log residuals, "
             "scene-shared scale; >0 = ctx16 better). disp_l = L2 of post-norm "
             "(shared depth_head.norm) patch-token difference ctx16 vs ctx4. "
             "right := err_ctx16 < err_ctx4 (strict); ties (mostly GT-invalid patches) "
             "excluded from AUC. GT enters ONLY via the Stage-1 err maps.",
             "",
             "## Spearman(disp_l, gain) per layer",
             "",
             "| scene | " + " | ".join(f"L{l}" for l in M.TAP_LAYERS) + " |",
             "|---|" + "---|" * len(M.TAP_LAYERS)]
    for scene in scenes:
        row = [r for r in all_spear if r["scene"] == scene]
        lines.append(f"| {scene} | " + " | ".join(
            _fmt(float(r["spearman"])) for r in row) + " |")
    mean_row = [_fmt(float(np.mean([float(r["spearman"]) for r in all_spear
                                    if r["layer"] == str(l) or int(r["layer"]) == l])))
                for l in M.TAP_LAYERS]
    lines.append("| **mean** | " + " | ".join(mean_row) + " |")

    # pooled spearman (raw concat) per layer
    lines += ["", "Pooled (all scenes concatenated): " +
              ", ".join(f"L{l} rho={_fmt(float(spearmanr(np.concatenate([scene_res[s][f'disp{l}'].ravel() for s in scenes]), np.concatenate([scene_res[s]['gain'].ravel() for s in scenes])).statistic))}" for l in M.TAP_LAYERS),
              "",
              "## Top-decile gain patches: share of total disp by layer",
              "",
              "| scene | " + " | ".join(f"L{l}" for l in M.TAP_LAYERS) + " |",
              "|---|" + "---|" * len(M.TAP_LAYERS)]
    for scene in scenes:
        sh = scene_res[scene]["top_decile_gain_disp_share"]
        lines.append(f"| {scene} | " + " | ".join(f"{v:.3f}" for v in sh) + " |")
    mean_sh = np.mean([scene_res[s]["top_decile_gain_disp_share"] for s in scenes], 0)
    lines.append("| **mean** | " + " | ".join(f"{v:.3f}" for v in mean_sh) + " |")

    feats = sorted({r["feature"] for r in all_auc})
    lines += ["", "## ROC-AUC of GT-free signals for the right/wrong label",
              "",
              "| feature | " + " | ".join(scenes) + " | **pooled** |",
              "|---|" + "---|" * (len(scenes) + 1)]
    pooled = {}
    for f in feats:
        per = []
        for scene in scenes:
            r = [r for r in all_auc if r["feature"] == f and r["scene"] == scene]
            per.append(float(r[0]["auc"]) if r else float("nan"))
        # pooled from concatenated per-scene signals
        sig_all, lab_all = [], []
        for scene in scenes:
            res = scene_res[scene]
            keep = ~res["tie"].ravel().astype(bool)
            lab_all.append(res["right"].ravel()[keep])
            sig_all.append(scene_signal(res, f)[keep])
        pooled[f] = roc_auc(np.concatenate(sig_all), np.concatenate(lab_all))
        lines.append(f"| {f} | " + " | ".join(_fmt(v) for v in per) +
                     f" | **{_fmt(pooled[f])}** |")

    best = max(feats, key=lambda f: (pooled[f] if np.isfinite(pooled[f]) else -1))
    best_layer = M.TAP_LAYERS[int(np.argmax([float(np.mean([float(r["spearman"]) for r in all_spear if int(r["layer"]) == l])) for l in M.TAP_LAYERS]))]
    excl = float(np.mean([float([r for r in all_auc if r["feature"] == "conf16"
                                 and r["scene"] == s][0]["frac_excluded"]) for s in scenes]))
    lines += ["", "## Conclusions", ""]
    lines.append(f"1. Layer concentration: the ctx4->ctx16 token displacement that best "
                 f"tracks the per-patch gain sits at **L{best_layer}** (mean Spearman vs "
                 f"gain, table above); top-decile-gain patches carry the disp shares shown "
                 f"(mean row).")
    lines.append(f"2. Best GT-free predictor of teacher-right: **{best}** "
                 f"(pooled AUC {_fmt(pooled[best])}; ties excluded = {excl:.1%} of patches, "
                 f"mostly GT-invalid).")
    lines.append("3. Implication for knowledge transfer: see per-layer correlation and "
                 "AUC tables - if disp at the late layers both tracks gain and predicts "
                 "right/wrong without GT, late-layer norm-space feature matching (the B5/C2 "
                 "family) is supervising exactly the tokens that long context moves; "
                 "conf x disp combos quantify how much of the oracle gate is recoverable "
                 "without GT.")
    out = os.path.join(args.out_dir, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines), pooled, best


def scene_signal(res, feature):
    """Reconstruct a pooled signal vector from the saved per-scene dict."""
    if feature == "conf16":
        return res["confp"].ravel()
    if feature.startswith("disp") and feature[4:].isdigit():
        return res[f"disp{feature[4:]}"].ravel()
    # z-scored combos recomputed per scene
    disp_stack = np.stack([zscore(res[f"disp{l}"].ravel()) for l in M.TAP_LAYERS], 0)
    disp_raw = np.stack([res[f"disp{l}"].ravel() for l in M.TAP_LAYERS], 0)
    conf_z = zscore(res["confp"].ravel())
    if feature == "disp_mean_z":
        return disp_stack.mean(0)
    if feature == "disp_consistency":
        return -(disp_raw.std(0) / (np.abs(disp_raw.mean(0)) + 1e-8))
    if feature == "fnorm16_23":
        return res["fnorm16_23"].ravel()
    if feature == "fnorm16_mean_z":
        return np.stack([zscore(res[f"fnorm16_{l}"].ravel()) for l in M.TAP_LAYERS], 0).mean(0)
    if feature == "conf_x_disp23":
        return conf_z * zscore(res["disp23"].ravel())
    if feature == "conf_x_disp_mean":
        return conf_z * disp_stack.mean(0)
    raise KeyError(feature)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--stage1_dir", default=STAGE1_DEFAULT)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--oom_retries", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "per_scene"), exist_ok=True)
    manifest = load_manifest(args.manifest)
    scenes = args.scenes or sorted(manifest["scenes"])
    progress(f"feature_traceback Stage2 启动 scenes={scenes}")

    vggt = M.load_teacher(args.device)
    all_spear, all_auc, scene_res = [], [], {}
    for scene in scenes:
        ev = manifest["scenes"][scene]["eval32_frames"]
        for attempt in range(args.oom_retries + 1):
            try:
                t0 = time.time()
                scene_data = get_scene_data(scene)
                res, spear_rows, auc_rows = run_scene(
                    vggt, scene, scene_data, ev, args.stage1_dir, args.device)
                scene_res[scene] = res
                all_spear += spear_rows
                all_auc += auc_rows
                np.savez_compressed(
                    os.path.join(args.out_dir, "per_scene", f"{scene}.npz"), **res)
                print(f"[scene {scene}] done in {time.time() - t0:.0f}s; "
                      f"spearman={[round(float(r['spearman']), 3) for r in spear_rows]}",
                      flush=True)
                del scene_data
                torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == args.oom_retries:
                    raise
                print(f"[scene {scene}] OOM, retry in 60s "
                      f"({attempt + 1}/{args.oom_retries})", flush=True)
                time.sleep(60)

    append = os.path.exists(os.path.join(args.out_dir, "spearman.csv"))
    with open(os.path.join(args.out_dir, "spearman.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPEAR_FIELDS)
        if not append:
            w.writeheader()
        w.writerows(all_spear)
    append = os.path.exists(os.path.join(args.out_dir, "auc.csv"))
    with open(os.path.join(args.out_dir, "auc.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUC_FIELDS)
        if not append:
            w.writeheader()
        w.writerows(all_auc)

    text, pooled, best = write_summary(args, scenes, all_spear, all_auc, scene_res)
    print(text, flush=True)
    progress(f"feature_traceback Stage2 完成；最佳无GT信号={best} "
             f"pooled AUC={pooled[best]:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
