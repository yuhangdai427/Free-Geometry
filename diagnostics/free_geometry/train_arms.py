#!/usr/bin/env python3
"""D3: 4-arm per-scene LoRA bake-off (VGGT x ScanNet++, 6 dev scenes).

Arms (identical everything except supervision):
  A1  deployed loss: PatchHuberCosine(all tokens, L4/11/17/23, 1.0+2.0cos)
      + CF-angle x2.0 + CF-distance(KL) x1.0  (the ACTUAL current loss)
  A2  PROJECTED patch-only MSE (depth_head.norm + projects[j], fp32)
  A3  output space: scale-invariant log-depth MSE vs teacher 8->4 depth (fp32)
  A4  camera/pose space: MSE vs teacher 8->4 pose enc (T + quat + FoV) (fp32)

Protocol: per-scene LoRA re-init (PEFT-faithful), 10 fixed train pairs x 10
epochs = 100 steps, batch 1, lr 3e-5 cosine + 15% warmup, wd 1e-5, clip 1.0,
AMP on (deployment-matching), camera token frozen, no augmentation, identical
pair order across arms. Teacher outputs cached once per scene (frozen model).

Probe eval (2 fixed pairs: E_depth, RAW/NORM/PROJECTED MSE, pose AUC) at steps
0/30/100. 32-view inference saved for steps 30/100 (+ baseline once).

Outputs:
  <run_root>/training_trace.csv
  <run_root>/probe_metrics.csv
  <run_root>/ckpts/<scene>/<arm>/step{N}_lora.pt(+_peft/)
  <run_root>/eval32/<exp>/model_results/scannetpp/<scene>/unposed/exports/...
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common
from common import STUDENT_INDICES, TAP_LAYERS, get_scene_data, gt_ixt_raw, load_manifest, stable_seed
import modeling as M
import abs_pose_loss as _apl
from abs_pose_loss import (loss_abs_pose_norm, loss_abs_pose_raw, loss_abs_t_fl,
                           loss_rkd_triplet_pose, loss_rkd_shared_pose,
                           loss_rkd_shared_pose_huber, loss_rkd_local_huber,
                           loss_rkd_mix_pose,
                           loss_xac_camtok, loss_xac2_camtok,
                           loss_scale_gauge, loss_couple16,
                           loss_rkd_ext_pose, loss_xap_pool,
                           loss_closure_align, loss_gram_pool)
from free_geometry.tta_v2 import (ControllerConfig, ProbeEvaluator, append_jsonl,
                                  edges_from_w2c, rot_edges_huber, select,
                                  tdir_cos_loss)

ARMS = ["A1_deployed", "A2_projected", "A3_outdepth", "A4_pose"]
FEATURE_ARMS = ["B1_norm_patch", "B2_deployed_camtok", "B3_raw_patch", "B4_norm_camtok"]
EPOCHS = 10
LR = 3e-5
ABS_POSE_W = 5.0  # SelfEvo camera-loss weight (config default.yaml:113)
CTK_W = 1.0  # camera-token KD weight for the *_CTK champion variants
             # (DA3 w=3.0 regressed the tails; start light, sweep if smoke is clean)
ABS_RAW_W = 1.0   # raw-target absolute pose loss weight (ABS-v2)
WARMUP_RATIO = 0.15
WD = 1e-5
CLIP = 1.0
EVAL_STEPS = (0, 30, 100)

# Protocol v2: --v2_couple_fix swaps abs_pose_loss.loss_couple (asymmetric valid
# masks: student side unmasked, teacher side conf-quantile-masked) for the
# symmetric form where BOTH depth sides take the mean over the SAME
# teacher-derived mask — identical semantics to losses.loss_couple_centers and
# tta_v2.couple_robust. Dispatch happens through this module-level name so all
# ~30 arm call sites pick the fix up without touching their code paths.
_V2_COUPLE_FIX = False
_loss_couple_orig = _apl.loss_couple

# Protocol v2: --v2_allpos keeps the masked INPUT but computes the maskdistill
# feature loss on ALL patch positions (w = teacher_patch_conf, no *patch_mask —
# equivalent to loss_b5_conf weighting). Read by loss_b5_maskdistill, so every
# arm routing its feature term through it (the whole C2M_* family and friends)
# switches together. Distinct from --loss_all_pos (which replaces pmask with
# ones in the training loop, so mask_ratio reports 1.0 and every other pmask
# consumer is affected).
_V2_ALLPOS = False


def loss_couple_sharedmask(pose_enc_s, depth_s, pose_enc_t_shared, depth4_t,
                           conf4_t=None, image_hw=_apl.IMAGE_HW):
    """Symmetric-valid-mask variant of abs_pose_loss.loss_couple (the
    --v2_couple_fix implementation). One mask, derived from the TEACHER depth
    (isfinite & >0 & conf >= 5% quantile), is applied to both the student and
    the teacher depth mean — the student no longer averages over its own
    (unmasked) valid set."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        def centers(pose):
            E, _ = pose_encoding_to_extri_intri(pose.float(), image_hw)
            R, t = E[..., :3, :3], E[..., :3, 3]
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)[0]

        d_t = depth4_t.squeeze(0).squeeze(-1).float() if depth4_t.dim() >= 4 \
            else depth4_t.float()
        m = torch.isfinite(d_t) & (d_t > 0)
        if conf4_t is not None:
            cf = conf4_t.squeeze(0).squeeze(-1).float() if conf4_t.dim() >= 4 \
                else conf4_t.float()
            q = torch.quantile(cf[m].flatten(), 0.05)
            m = m & (cf >= q)

        def stat(pose, depth):
            c = centers(pose)
            spr = (c - c.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()
            d = depth.squeeze(0).squeeze(-1).float() if depth.dim() >= 4 \
                else depth.float()
            md = d[m].mean().clamp_min(1e-6)
            return torch.log(spr.clamp_min(1e-6)) - torch.log(md)

        cs = stat(pose_enc_s, depth_s)
        ct = stat(pose_enc_t_shared, depth4_t).detach()
        loss = (cs - ct) ** 2
    return loss.squeeze(), {"couple": float(loss)}


def loss_couple(*args, **kwargs):
    if _V2_COUPLE_FIX:
        return loss_couple_sharedmask(*args, **kwargs)
    return _loss_couple_orig(*args, **kwargs)


# --------------------------------------------------------------------------
# protocol v2 wiring (GT-free rel branch, grad cap, probe, ckpt)
# --------------------------------------------------------------------------

def v2_rel_pose_loss(pose_enc_s, teacher_cache, image_hw, weight):
    """v2 robust relative-pose branch: rot_edges_huber + tdir_cos_loss on the
    student shared views vs the DETACHED cached teacher shared views, both in
    the corrected w2c convention (T_{i<-j} = E_i @ inv(E_j), same as
    losses.loss_rel_pose). weight multiplies (rot + tdir). Student pose_enc may
    carry extra views (SS8M: 8 at STUDENT_INDICES; XRKD: shared first 4)."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    pe = pose_enc_s
    if pe.shape[1] > 4:
        pe = pe[:, STUDENT_INDICES] if pe.shape[1] == 2 * len(STUDENT_INDICES) \
            else pe[:, :4]
    with torch.autocast(device_type="cuda", enabled=False):
        ext_s, _ = pose_encoding_to_extri_intri(pe.float(), image_hw,
                                                pose_encoding_type="absT_quaR_FoV")
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        ext_t, _ = pose_encoding_to_extri_intri(pt, image_hw,
                                                pose_encoding_type="absT_quaR_FoV")
        e_s = edges_from_w2c(ext_s[0])
        e_t = edges_from_w2c(ext_t[0])
        rot = rot_edges_huber(e_s["R_rel"], e_t["R_rel"].detach())
        tdir = tdir_cos_loss(e_s["t_rel"], e_t["t_rel"].detach(), e_t["baseline"])
        loss = weight * (rot["loss"] + tdir["loss"])
    return loss, {"v2_rel": float(loss), "v2_rel_rot": float(rot["loss"]),
                  "v2_rel_tdir": float(tdir["loss"])}


def v2_grad_cap_backward(loss_base, loss_rel, params, scaler, state):
    """GradScaler-exact split backward with a per-param cap on the rel branch.

    Implementation: TWO ordinary fused backwards instead of autograd.grad —
    bit-exactness matters: autograd.grad on an autocast-built graph computes
    weight grads behind the autocast boundary with different precision rules
    (measured: conv-weight grad off by >30% on a toy autocast model), while
    backward(retain_graph=True) + backward() is bitwise identical to a single
    fused backward. Each branch is scaled by scaler.scale() exactly like the
    normal path; g_base is snapshotted between the two backwards, g_R is
    per-param-norm-capped (the cap factor is scale-invariant, applied in
    scaled space), then merged back into p.grad so the existing
    scaler.unscale_ -> clip_grad_norm_ -> scaler.step -> scaler.update chain
    runs UNCHANGED (inf detection / step skipping behave as usual).

    state: dict, mutated in place; keys "medians" (per-update median of
    per-param unscaled ||g_R|| over the first 10 updates), "C_R" (float or
    None until calibrated: C_R = 4 * median(medians), then constant).
    Returns trace extras."""
    for p in params:
        p.grad = None
    scaler.scale(loss_base).backward(retain_graph=True)
    g_base = [p.grad.clone() if p.grad is not None else None for p in params]
    for p in params:
        p.grad = None
    scaler.scale(loss_rel).backward()
    with torch.no_grad():
        s = scaler.get_scale()
        g_base_norm = 0.0
        for g in g_base:
            if g is not None:
                g_base_norm += float((g * g).sum()) / (s * s)
        g_base_norm = g_base_norm ** 0.5
        per = [float((p.grad * p.grad).sum()) ** 0.5 / s
               for p in params if p.grad is not None]
        g_R_norm = float(sum(v * v for v in per) ** 0.5) if per else 0.0
        if len(state["medians"]) < 10:
            if per:
                state["medians"].append(float(np.median(per)))
            if len(state["medians"]) == 10:
                state["C_R"] = 4.0 * float(np.median(state["medians"]))
        C_R = state["C_R"]
        if C_R is not None:
            for p in params:
                if p.grad is None:
                    continue
                # per-param norm cap in unscaled space, applied to the scaled
                # grad (factor is scale-invariant; no host sync per param)
                f = torch.clamp(C_R * s / ((p.grad * p.grad).sum().sqrt() + 1e-12),
                                max=1.0)
                p.grad.mul_(f)
        for p, gb in zip(params, g_base):
            if gb is None and p.grad is None:
                continue
            total = p.grad if p.grad is not None else 0.0
            if gb is not None:
                total = total + gb
            p.grad = total
    return {"g_base_norm": g_base_norm, "g_R_norm": g_R_norm,
            "C_R": float(C_R) if C_R is not None else -1.0}


def build_v2_probe_contexts(scene, sc, probe_caches, probe_images4, base):
    """One ProbeEvaluator context per manifest probe pair, built from the SAME
    teacher caches the GT probe uses (GT-free quantities only: cached teacher
    features/depth/conf/poses + teacher-conf valid mask). Returns
    (contexts, overlaps) with overlaps[i] True when probe pair i collides with
    a train pair on teacher_frames or student_frames keys."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    train_t8 = {tuple(p["teacher_frames"]) for p in sc["train_pairs"]}
    train_s4 = {tuple(p["student_frames"]) for p in sc["train_pairs"]}
    contexts, overlaps = [], []
    with torch.no_grad():
        for pi, (pair, cache, im4) in enumerate(zip(
                sc["probe_pairs"], probe_caches, probe_images4)):
            ph, pw = cache["patch_hw"]
            H, W = im4.shape[-2:]
            with torch.autocast(device_type="cuda", enabled=False):
                pt = cache["pose_enc8"][:, STUDENT_INDICES].float()
                ext_t, _ = pose_encoding_to_extri_intri(
                    pt, (H, W), pose_encoding_type="absT_quaR_FoV")
                ext_t = ext_t[0].detach()                     # [4,3,4] w2c
                R, t = ext_t[..., :3, :3], ext_t[..., :3, 3]
                centers_t = (-R.transpose(-1, -2)
                             @ t.unsqueeze(-1)).squeeze(-1).detach()
                conf = cache["conf4"].float()
                if conf.dim() == 5:
                    conf = conf.squeeze(2)
                conf = conf[0]                                # [4,H,W]
                thr = torch.quantile(conf.flatten(), 0.05)
                valid_t = (torch.isfinite(conf) & (conf >= thr)).detach()
                depth_t = cache["depth4"].squeeze(0).squeeze(-1).float().detach()
                feats_t = {l: M.to_norm(
                    base.depth_head,
                    M.to_patch(cache["feats"][l][:, STUDENT_INDICES].float()),
                ).squeeze(0).detach() for l in TAP_LAYERS}    # {l: [4,P,C]}
                feat_w = teacher_patch_conf(cache, (ph, pw))[0]
            contexts.append({
                "pair_id": f"probe{pi}", "scene": scene,
                "patch_grid": (ph, pw), "mask_ratio": 0.5,
                "images_meta": im4,
                "teacher": {"features": feats_t, "feat_w": feat_w,
                            "ext_w2c": ext_t, "centers": centers_t,
                            "depth": depth_t, "valid": valid_t},
            })
            overlaps.append(tuple(pair["teacher_frames"]) in train_t8
                            or tuple(pair["student_frames"]) in train_s4)
    return contexts, overlaps


def make_v2_probe_forward(student, base):
    """forward_fn(images4, mask[S,ph,pw] bool) for ProbeEvaluator.evaluate.
    Pixel fill matches the training-time masking convention exactly
    (mask_image_blocks: multiply [0,1] images by (1 - block), i.e. ZERO fill).
    The forward runs in fp32 with no autocast, matching evaluate_probes /
    infer_eval32 (deployment) numerics. Runs under the caller's no_grad."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    def fwd(images_meta, mask):
        images4 = images_meta
        ph, pw = mask.shape[-2], mask.shape[-1]
        H, W = images4.shape[-2:]
        m = mask.to(images4.device)
        m = m.repeat_interleave(H // ph, dim=1).repeat_interleave(W // pw, dim=2)
        masked = images4 * (~m)[:, None].float()
        feats24, psi, preds = M.student_preds(student, masked)
        feats_d = {l: M.to_norm(base.depth_head,
                                M.to_patch(feats24[l].float()))[0]
                   for l in TAP_LAYERS}
        with torch.autocast(device_type="cuda", enabled=False):
            ext, _ = pose_encoding_to_extri_intri(
                preds["pose_enc"].float(), (H, W),
                pose_encoding_type="absT_quaR_FoV")
        ext = ext[0]                                          # [4,3,4] w2c
        R, t = ext[..., :3, :3], ext[..., :3, 3]
        centers = (-R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
        depth = preds["depth"].squeeze(0).squeeze(-1).float()
        return {"features": feats_d, "ext_w2c": ext, "centers": centers,
                "depth": depth}

    return fwd


def run_v2_probe(evaluator, fwd, scene, arm, step, run_root, overlaps,
                 n_train_pairs, log_fn=None):
    """Evaluate + append one JSONL line per (pair, mask) record to
    <run_root>/probe_trace/<scene>.jsonl. log_fn(out) — when given — receives
    the evaluate() result for swanlab reporting (x-axis = real step)."""
    out = evaluator.evaluate(step, fwd)
    path = os.path.join(run_root, "probe_trace", f"{scene}.jsonl")
    for r in out["records"]:
        pi = int(r["pair_id"].replace("probe", ""))
        append_jsonl(path, {
            "scene": scene, "arm": arm, "step": int(step),
            "pair_id": r["pair_id"], "mask_id": int(r["mask_id"]),
            "components": r["components"], "total": float(r["total"]),
            "probe_train_overlap": bool(overlaps[pi]),
            "n_train_pairs": int(n_train_pairs),
        })
    if log_fn is not None:
        try:
            log_fn(out)
        except Exception as e:
            print(f"[{scene}] v2 probe logging failed ({e}); continuing",
                  flush=True)
    return out


def save_v2_ckpt(student, run_root, scene, arm, step):
    """LoRA save for protocol v2 checkpoint selection. HARD assert on success:
    save_lora_weights writes a _peft adapter dir (and the .pt only when the
    camera token is trainable), so the _peft dir is the payload we verify."""
    ckpt_dir = os.path.join(run_root, "ckpts", scene, arm, "v2")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step{step}_lora.pt")
    student.save_lora_weights(path)
    peft_dir = path.replace(".pt", "_peft")
    assert os.path.isdir(peft_dir) and len(os.listdir(peft_dir)) > 0, \
        f"v2 ckpt save failed: {peft_dir} missing or empty"
    return path


# --------------------------------------------------------------------------
# swanlab reporting (protocol v2 monitoring). All helpers are no-ops when
# swanlab is disabled or unavailable, and EVERY call is exception-guarded:
# logging must never kill a training run.
# --------------------------------------------------------------------------

def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and v == v and abs(v) != float("inf")


def _swan_safe_init(args, scene, arm):
    """One swanlab experiment per (scene, arm), named {scene}_{arm}{suffix}.
    Returns the Run or None (disabled / init failure — training continues)."""
    if not getattr(args, "swanlab", False):
        return None
    try:
        import swanlab
        kw = dict(
            project=args.swanlab_project,
            name=f"{scene}_{arm}{args.swanlab_suffix}",
            description=f"{common._CURRENT_DATASET} | {arm} | "
                        f"run_root={args.run_root} | seed={args.seed}",
            config={k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool)) or v is None},
        )
        if getattr(args, "swanlab_mode", None):
            kw["mode"] = args.swanlab_mode
        if getattr(args, "swanlab_logdir", None):
            kw["log_dir"] = args.swanlab_logdir
        return swanlab.init(**kw)
    except Exception as e:  # swanlab missing / offline / auth failure ...
        print(f"[{scene}] swanlab init failed ({e}); continuing unlogged",
              flush=True)
        return None


def _swan_log(run, metrics, step=None):
    """Exception-guarded run.log; non-finite values are dropped."""
    if run is None or not metrics:
        return
    try:
        clean = {k: float(v) for k, v in metrics.items() if _finite(v)}
        if clean:
            run.log(clean, step=step)
    except Exception as e:
        print(f"[swanlab] log failed ({e}); continuing", flush=True)


def _swan_finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


def _pair_metrics(pi, loss, extra) -> Dict[str, float]:
    """Per-training-pair curves (x-axis = global step, so pair-to-pair spread
    is visible against the shared time axis). Keys: pair/p{pi}/{total,feature,
    rkd,couple,rel,v2_rel} — only the components present in this arm's extra."""
    e = extra or {}
    m = {f"pair/p{pi}/total": float(loss)}
    if _finite(e.get("feat")):
        m[f"pair/p{pi}/feature"] = float(e["feat"])
    if _finite(e.get("rkd")):
        m[f"pair/p{pi}/rkd"] = float(e["rkd"])
    if _finite(e.get("couple")):
        m[f"pair/p{pi}/couple"] = float(e["couple"])
    if _finite(e.get("rel_rot")) and _finite(e.get("rel_tdir")):
        m[f"pair/p{pi}/rel"] = float(e["rel_rot"]) + float(e["rel_tdir"])
    if _finite(e.get("v2_rel")):
        m[f"pair/p{pi}/v2_rel"] = float(e["v2_rel"])
    return m


def _probe_metrics(out) -> Dict[str, float]:
    """Swanlab dict for one ProbeEvaluator.evaluate() result: per
    (pair, mask) components + probe/mean/* over records. x-axis = out['step']."""
    m: Dict[str, float] = {}
    comps: Dict[str, List[float]] = {}
    totals: List[float] = []
    for r in out["records"]:
        j = int(str(r["pair_id"]).replace("probe", ""))
        k = int(r["mask_id"])
        for c, v in r["components"].items():
            if _finite(v):
                m[f"probe/pair{j}/mask{k}/{c}"] = float(v)
                comps.setdefault(c, []).append(float(v))
        if _finite(r["total"]):
            m[f"probe/pair{j}/mask{k}/total"] = float(r["total"])
            totals.append(float(r["total"]))
    for c, vals in comps.items():
        m[f"probe/mean/{c}"] = sum(vals) / len(vals)
    if totals:
        m["probe/mean/total"] = sum(totals) / len(totals)
    return m


def _swan_selector_summary(run_root, scene, arm, run):
    """Offline checkpoint-selection replay: read this (scene, arm)'s records
    from probe_trace/<scene>.jsonl and run tta_v2.controller.select on them,
    then log the summary. No-op when the trace/step-0 baseline is absent or
    anything fails."""
    if run is None:
        return
    path = os.path.join(run_root, "probe_trace", f"{scene}.jsonl")
    if not os.path.exists(path):
        return
    try:
        recs = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("arm") == arm and r.get("scene") == scene:
                    recs.append(r)
        trace = []
        for step in sorted({int(r["step"]) for r in recs}):
            trace.append({"step": step, "records": [
                {"pair_id": r["pair_id"], "mask_id": r["mask_id"],
                 "components": r["components"], "total": r["total"]}
                for r in recs if int(r["step"]) == step]})
        if not trace or int(trace[0]["step"]) != 0:
            return
        out = select(trace, ControllerConfig())
        _swan_log(run, {
            "selector/selected_step": float(out["selected_step"]),
            "selector/fell_back_to_baseline": float(out["fell_back_to_baseline"]),
            "selector/improvement": float(out["improvement"]),
        }, step=int(trace[-1]["step"]))
        print(f"[{scene}] {arm} selector replay: selected_step="
              f"{out['selected_step']} fell_back={out['fell_back_to_baseline']} "
              f"improvement={out['improvement']:.4f}", flush=True)
    except Exception as e:
        print(f"[{scene}] selector replay failed ({e}); continuing", flush=True)


def _metrics_of(entry):
    """Extract (auc03, f1) from a baselines-json scene entry, tolerating
    key spelling variants."""
    if not isinstance(entry, dict):
        return None, None
    auc = entry.get("auc03", entry.get("auc3"))
    f1 = entry.get("f1", entry.get("recon_fscore"))
    return auc, f1


def _swan_scene_eval(args, scene, arm, run):
    """Scene-level baseline-vs-TTA eval comparison.

    baselines.json schema (nested): {model: {arm: {scene: {auc03, f1, ...}}}}
    with "_" -prefixed meta keys allowed at the scene level. Reads model=vggt,
    arms "baseline" (reference) and the current arm (TTA side). Logs
    eval/{baseline_auc03,baseline_f1,auc03,f1,d_auc03_rel,d_f1_rel}; when the
    current-arm entry is unavailable, drops <run_root>/scene_eval_pending.json
    so an external eval script can backfill (run_eval.py runs outside this
    process — per-scene recon AUC/F1 is NOT computable in-train_arms)."""
    base_e = arm_e = None
    try:
        with open(args.v2_baselines_json) as f:
            d = json.load(f)
        vggt = d.get("vggt", {})

        def _scene_entry(arm_name):
            a = vggt.get(arm_name, {})
            scenes = {k: v for k, v in a.items()
                      if isinstance(v, dict) and not k.startswith("_")}
            return scenes.get(scene)

        base_e = _scene_entry("baseline")
        arm_e = _scene_entry(arm)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[{scene}] baselines json unreadable ({e}); pending eval",
              flush=True)
    b_auc, b_f1 = _metrics_of(base_e)
    t_auc, t_f1 = _metrics_of(arm_e)
    m = {}
    if _finite(b_auc):
        m["eval/baseline_auc03"] = float(b_auc)
    if _finite(b_f1):
        m["eval/baseline_f1"] = float(b_f1)
    if _finite(t_auc):
        m["eval/auc03"] = float(t_auc)
    if _finite(t_f1):
        m["eval/f1"] = float(t_f1)
    if _finite(b_auc) and _finite(t_auc) and b_auc:
        m["eval/d_auc03_rel"] = (float(t_auc) - float(b_auc)) / abs(float(b_auc))
    if _finite(b_f1) and _finite(t_f1) and b_f1:
        m["eval/d_f1_rel"] = (float(t_f1) - float(b_f1)) / abs(float(b_f1))
    if m:
        _swan_log(run, m)
        print(f"[{scene}] {arm} eval: {m}", flush=True)
    else:
        _append_pending_eval(args, scene, arm)


def _append_pending_eval(args, scene, arm):
    """Record an unfinished scene-level eval comparison for the external
    backfill script (run_eval.py aggregates recon metrics outside this
    process)."""
    try:
        path = os.path.join(args.run_root, "scene_eval_pending.json")
        items = []
        if os.path.exists(path):
            with open(path) as f:
                items = json.load(f)
        items = [i for i in items
                 if not (i.get("scene") == scene and i.get("arm") == arm)]
        items.append({
            "scene": scene, "arm": arm,
            "run_root": os.path.abspath(args.run_root),
            "ckpt_dir": os.path.abspath(
                os.path.join(args.run_root, "ckpts", scene, arm)),
            "dataset": common._CURRENT_DATASET,
            "ts": time.time(),
        })
        with open(path, "w") as f:
            json.dump(items, f, indent=1)
    except Exception as e:
        print(f"[{scene}] pending-eval record failed ({e}); continuing",
              flush=True)


def import_deployed_losses():
    spec = importlib.util.spec_from_file_location(
        "train_vggt", os.path.join(_HERE, "..", "..", "scripts", "train_vggt.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DeployedArm:
    """A1: exact deployed objective on cached teacher outputs."""

    def __init__(self):
        mod = import_deployed_losses()
        self.feat = mod.VGGTPatchHuberCosineLoss(
            student_frame_indices=STUDENT_INDICES, target_layers=TAP_LAYERS,
            huber_weight=1.0, cos_weight=2.0, delta=1.0)
        self.cf_angle = mod.VGGTCrossFrameCFAngleLoss(
            student_frame_indices=STUDENT_INDICES, num_teacher_views=8, target_layer=23,
            topk=4, num_ref_samples=256, num_shared_samples=256,
            angle1_weight=1.0, angle2_weight=1.0, angle3_weight=1.0,
            shared_chunk_size=64, selection_mode="mixed")
        self.cf_dist = mod.VGGTCrossFrameCFDistanceLoss(
            student_frame_indices=STUDENT_INDICES, num_teacher_views=8, target_layer=23,
            topk=4, num_ref_samples=256, num_shared_samples=256,
            d1_weight=1.0, d2_weight=1.0, d3_weight=1.0,
            shared_chunk_size=64, distance_chunk_size=16, distance_type="l2",
            normalize_distance=True, temperature=1.0, distance_mode="kl",
            huber_beta=0.5, selection_mode="mixed")
        self._wrap = mod.VGGGTFreeGeometryOutput if hasattr(mod, "VGGGTFreeGeometryOutput") else None
        from vggt.vggt.test_time_adaption.models import VGGTFreeGeometryOutput
        self._out_cls = VGGTFreeGeometryOutput

    def _wrap_out(self, layer_feats: Dict[int, torch.Tensor]):
        return self._out_cls(layer_features=layer_feats, frame_features={},
                             global_features={}, camera_tokens={})

    def cf_terms(self, teacher_cache, feats24_s) -> torch.Tensor:
        """The deployed CF-angle x2 + CF-distance x1 terms on full tokens."""
        t_out = self._wrap_out(teacher_cache["feats"])
        s_out = self._wrap_out({l: feats24_s[l] for l in TAP_LAYERS})
        cf_a, _ = self.cf_angle(t_out, s_out)
        cf_d, _ = self.cf_dist(t_out, s_out)
        return 2.0 * cf_a + 1.0 * cf_d, {"cf_angle": float(cf_a), "cf_dist": float(cf_d)}

    def __call__(self, teacher_cache, feats24_s) -> torch.Tensor:
        t_out = self._wrap_out(teacher_cache["feats"])
        s_out = self._wrap_out({l: feats24_s[l] for l in TAP_LAYERS})
        feat_loss, _ = self.feat(t_out, s_out)
        cf_loss, extra = self.cf_terms(teacher_cache, feats24_s)
        return feat_loss + cf_loss, {"feat": float(feat_loss), **extra}


def loss_a2_projected(depth_head, teacher_cache, feats24_s, patch_hw) -> torch.Tensor:
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_patch(feats24_s[layer].float())
        ht = M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float())
        zs = M.to_projected(depth_head, hs, layer, patch_hw)
        zt = M.to_projected(depth_head, ht, layer, patch_hw)
        total = total + torch.mean((zs - zt) ** 2)
    return total / len(TAP_LAYERS), {}


def loss_a3_outdepth(teacher_cache, preds) -> torch.Tensor:
    depth = preds["depth"].squeeze(0).squeeze(-1).float()       # [4,H,W]
    tdepth = teacher_cache["depth4"].squeeze(0).squeeze(-1).float()
    valid = torch.isfinite(tdepth) & (tdepth > 0)
    r = torch.log(depth.clamp_min(1e-6)) - torch.log(tdepth.clamp_min(1e-6))
    r = torch.where(valid, r, torch.zeros_like(r))
    n = valid.sum().clamp_min(1)
    c = (r * valid).sum() / n
    return ((((r - c) ** 2) * valid).sum() / n), {}


def loss_a4_pose(teacher_cache, preds) -> torch.Tensor:
    ps = preds["pose_enc"].float()                                   # [1,4,9]
    pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()      # [1,4,9]
    t_s, q_s, f_s = ps[..., :3], ps[..., 3:7], ps[..., 7:]
    t_t, q_t, f_t = pt[..., :3], pt[..., 3:7], pt[..., 7:]
    huber = torch.nn.functional.smooth_l1_loss
    l_t = huber(t_s, t_t)
    q_s_n = torch.nn.functional.normalize(q_s, dim=-1)
    q_t_n = torch.nn.functional.normalize(q_t, dim=-1)
    l_q = (1.0 - (q_s_n * q_t_n).sum(-1).abs()).mean()
    l_f = huber(f_s, f_t)
    return l_t + l_q + l_f, {"pose_T": float(l_t), "pose_q": float(l_q), "pose_fov": float(l_f)}


def _huber_cos(hs: torch.Tensor, ht: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Deployed patch-loss FORM (SmoothL1 beta=1 w=1 + 2*(1-cos)).
    hs/ht: [B,S,P,C]; w: optional per-patch weight [B,S,P] normalized to mean 1."""
    if w is None:
        huber = torch.nn.functional.smooth_l1_loss(hs, ht, beta=1.0)
        cos = torch.nn.functional.cosine_similarity(hs, ht, dim=-1).mean()
        return huber + 2.0 * (1.0 - cos)
    huber_t = torch.nn.functional.smooth_l1_loss(hs, ht, beta=1.0, reduction="none").mean(dim=-1)
    cos_t = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
    huber = (huber_t * w).mean()
    cos = (cos_t * w).mean()
    return huber + 2.0 * (1.0 - cos)


def loss_b5_margin(depth_head, teacher_cache, feats24_s, patch_hw, margin=0.85):
    """B5 + tolerance margin (Depth Anything V1 style): patches whose
    teacher-student cosine already exceeds `margin` exit the loss - don't
    spend LoRA capacity re-copying what is already aligned."""
    w = teacher_patch_conf(teacher_cache, patch_hw)
    total, kept = 0.0, 0.0
    for layer in TAP_LAYERS:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
        cos_t = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
        keep = (cos_t < margin).float()
        wl = w * keep
        kept += float(keep.mean())
        huber_t = torch.nn.functional.smooth_l1_loss(hs, ht, beta=1.0, reduction="none").mean(dim=-1)
        denom = wl.sum().clamp_min(1e-8)
        total = total + (huber_t * wl).sum() / denom + 2.0 * (1.0 - (cos_t * wl).sum() / denom)
    return total / len(TAP_LAYERS), {"keep_frac": kept / len(TAP_LAYERS)}


def _feat_mask_hook(pmask):
    """Forward hook on aggregator.patch_embed: zero masked patch tokens AFTER
    patch embedding (MGD/A2MIM-style feature-level masking). pmask [1,S,P],
    1 = masked."""
    def hook(module, inp, out):
        B, S, P = pmask.shape
        m = (1.0 - pmask.reshape(B * S, P, 1))
        if isinstance(out, dict):
            t = out["x_norm_patchtokens"]
            out["x_norm_patchtokens"] = t * m.to(t.device, t.dtype)
            return out
        return out * m.to(out.device, out.dtype)
    return hook


def mask_per_view_asym(images, patch_hw, gen, hi=0.9, lo=0.1):
    """CroCo/SiamMAE-style per-view asymmetric masking: one random view masked
    hi (90%), the rest lo (10%) -> forces cross-view completion.
    Returns (masked_images, patch_mask[1,S,P], 1=masked). gen must be CPU."""
    B, S, C, H, W = images.shape
    ph, pw = patch_hw
    mask = (torch.rand(S, ph * pw, generator=gen) < lo).float()
    j = int(torch.randint(S, (1,), generator=gen))
    mask[j] = (torch.rand(ph * pw, generator=gen) < hi).float()
    mask = mask.to(images.device)
    m = mask.reshape(S, ph, pw).repeat_interleave(H // ph, 1).repeat_interleave(W // pw, 2)
    out = images * (1.0 - m)[:, None]
    return out, mask.reshape(1, S, ph * pw)


def loss_ctm_camtok_j(camera_head, teacher_cache, feats24_s, j):
    """CamTokHC on ONE masked view j only (L23, post token_norm): the student
    must reconstruct view j's camera token without its input anchor."""
    with torch.autocast(device_type="cuda", enabled=False):
        cs = camera_head.token_norm(feats24_s[23][:, j, 0, :].float())
        ct = camera_head.token_norm(
            teacher_cache["feats"][23][:, STUDENT_INDICES[j], 0, :].float()).detach()
        huber = torch.nn.functional.smooth_l1_loss(cs, ct, beta=1.0)
        cos = torch.nn.functional.cosine_similarity(cs, ct, dim=-1).mean()
        loss = huber + 2.0 * (1.0 - cos)
        return loss, {"ctm_camtok": float(loss)}


def loss_pose_rel_confp(pose_s: torch.Tensor, pose_t: torch.Tensor,
                        conf4: torch.Tensor):
    """rel-pose weighted by the teacher's depth confidence on the two views of
    each pair (high-conf views => more reliable relative-pose target)."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        H, W = 378, 504
        ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
        ext_t, _ = pose_encoding_to_extri_intri(pose_t.float(), (H, W))
        R_s, t_s = ext_s[..., :3, :3], ext_s[..., :3, 3]
        R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]
        S = R_s.shape[1]
        cf = conf4.float()
        if cf.dim() == 5:
            cf = cf.squeeze(2)
        vw = cf.mean(dim=(2, 3)).detach()          # [1,S] per-view mean conf
        vw = vw / vw.mean().clamp_min(1e-8)
        terms, ws = [], []
        for i in range(S):
            for j in range(i + 1, S):
                Rr_s = R_s[:, i].transpose(-1, -2) @ R_s[:, j]
                Rr_t = R_t[:, i].transpose(-1, -2) @ R_t[:, j]
                rot = ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
                tr_s = R_s[:, i].transpose(-1, -2) @ (t_s[:, j] - t_s[:, i])[..., None]
                tr_t = R_t[:, i].transpose(-1, -2) @ (t_t[:, j] - t_t[:, i])[..., None]
                tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
                tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
                tdir = (1.0 - (tn_s * tn_t).sum(-1)).mean()
                terms.append(rot + tdir)
                ws.append(vw[0, i] * vw[0, j])
        r = torch.stack(terms)
        w = torch.stack(ws)
        w = w / w.mean().clamp_min(1e-8)
        loss = (w * r).sum() / w.sum()
        return loss, {"rel_confp": float(loss)}


def rel_rot_deg(pose_s: torch.Tensor, pose_t: torch.Tensor) -> float:
    """Mean relative-rotation disagreement between two trajectories, degrees."""
    import math
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    with torch.autocast(device_type="cuda", enabled=False):
        ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (378, 504))
        ext_t, _ = pose_encoding_to_extri_intri(pose_t.float(), (378, 504))
        Rs = ext_s[0, :, :3, :3]
        Rt = ext_t[0, :, :3, :3]
        S = Rs.shape[0]
        vals = []
        for i in range(S):
            for j in range(i + 1, S):
                Rr_s = Rs[i].transpose(-1, -2) @ Rs[j]
                Rr_t = Rt[i].transpose(-1, -2) @ Rt[j]
                chord = ((Rr_s - Rr_t) ** 2).sum().clamp_min(1e-12).sqrt().item()
                vals.append(2.0 * math.degrees(math.asin(min(1.0, chord / (2 * math.sqrt(2.0))))))
        return float(sum(vals) / max(1, len(vals)))


def loss_pose_rel_scale(pose_s: torch.Tensor, pose_t: torch.Tensor):
    """rel-pose + translation SCALE ratio: (log|t_s| - log|t_t|)^2 per pair.
    The direction-only rel-pose is scale-free by design, but TSDF fusion needs
    cross-frame scale consistency - the magnitude term attacks that directly."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        H, W = 378, 504
        ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
        ext_t, _ = pose_encoding_to_extri_intri(pose_t.float(), (H, W))
        R_s, t_s = ext_s[..., :3, :3], ext_s[..., :3, 3]
        R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]
        S = R_s.shape[1]
        rot_terms, tdir_terms, scl_terms = [], [], []
        for i in range(S):
            for j in range(i + 1, S):
                Rr_s = R_s[:, i].transpose(-1, -2) @ R_s[:, j]
                Rr_t = R_t[:, i].transpose(-1, -2) @ R_t[:, j]
                rot_terms.append(((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean())
                tr_s = R_s[:, i].transpose(-1, -2) @ (t_s[:, j] - t_s[:, i])[..., None]
                tr_t = R_t[:, i].transpose(-1, -2) @ (t_t[:, j] - t_t[:, i])[..., None]
                tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
                tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
                tdir_terms.append((1.0 - (tn_s * tn_t).sum(-1)).mean())
                ls = torch.log(tr_s.squeeze(-1).norm(dim=-1).clamp_min(1e-4))
                lt = torch.log(tr_t.squeeze(-1).norm(dim=-1).clamp_min(1e-4)).detach()
                scl_terms.append(((ls - lt) ** 2).mean())
        rot = torch.stack(rot_terms).mean()
        tdir = torch.stack(tdir_terms).mean()
        scl = torch.stack(scl_terms).mean()
        return rot + tdir + scl, {"rel_rot": float(rot), "rel_tdir": float(tdir),
                                  "rel_scale": float(scl)}


def loss_pose_cycle(pose_s: torch.Tensor):
    """Teacher-free cycle closure on the student's 4 views: for each triangle
    (i,j,k), composed relative transform i->j->k->i must be identity.
    Rotation chordal to I + normalized composed translation magnitude."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        H, W = 378, 504
        ext, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
        R, t = ext[0, ..., :3, :3], ext[0, ..., :3, 3]  # [S,3,3],[S,3]
        S = R.shape[0]
        terms = []
        I3 = torch.eye(3, device=R.device, dtype=R.dtype)
        import itertools
        for i, j, k in itertools.permutations(range(S), 3):
            R_ij = R[i].transpose(-1, -2) @ R[j]
            t_ij = R[i].transpose(-1, -2) @ (t[j] - t[i])
            R_jk = R[j].transpose(-1, -2) @ R[k]
            t_jk = R[j].transpose(-1, -2) @ (t[k] - t[j])
            R_ki = R[k].transpose(-1, -2) @ R[i]
            t_ki = R[k].transpose(-1, -2) @ (t[i] - t[k])
            R_cyc = R_ki @ R_jk @ R_ij
            t_cyc = (R_ki @ R_jk @ t_ij.unsqueeze(-1)).squeeze(-1) \
                + (R_ki @ t_jk.unsqueeze(-1)).squeeze(-1) + t_ki
            rot = ((R_cyc - I3) ** 2).sum()
            tnorm = t_cyc.norm() / (t_ij.norm() + t_jk.norm() + t_ki.norm() + 1e-8)
            terms.append(rot + tnorm)
        loss = torch.stack(terms).mean()
        return loss, {"cyc": float(loss)}


def loss_pose_rel_hard(pose_s: torch.Tensor, pose_t: torch.Tensor, hard_w: float = 2.0):
    """Hard-pair weighted rel-pose: per-pair residuals (student vs teacher),
    pairs at/above the median residual get hard_w x weight. GT-free OHEM."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        H, W = 378, 504
        ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
        ext_t, _ = pose_encoding_to_extri_intri(pose_t.float(), (H, W))
        R_s, t_s = ext_s[..., :3, :3], ext_s[..., :3, 3]
        R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]
        S = R_s.shape[1]
        terms = []
        for i in range(S):
            for j in range(i + 1, S):
                Rr_s = R_s[:, i].transpose(-1, -2) @ R_s[:, j]
                Rr_t = R_t[:, i].transpose(-1, -2) @ R_t[:, j]
                rot = ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
                tr_s = R_s[:, i].transpose(-1, -2) @ (t_s[:, j] - t_s[:, i])[..., None]
                tr_t = R_t[:, i].transpose(-1, -2) @ (t_t[:, j] - t_t[:, i])[..., None]
                tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
                tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
                tdir = (1.0 - (tn_s * tn_t).sum(-1)).mean()
                terms.append(rot + tdir)
        r = torch.stack(terms)
        w = torch.where(r.detach() >= r.detach().median(),
                        torch.full_like(r, hard_w), torch.ones_like(r))
        loss = (w * r).sum() / w.sum()
        return loss, {"rel_hard": float(loss), "rel_hard_maxw": float(w.max())}


def loss_pose_rel_cap(pose_s: torch.Tensor, pose_t: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    """loss_pose_rel with a per-pair Huber cap on the rotation chordal error:
    outlier view pairs (where the teacher's own relative pose is bad) get
    bounded gradients instead of dominating the mean. Translation-direction
    term unchanged (already bounded in [0,2])."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    H, W = 378, 504
    ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
    ext_t, _ = pose_encoding_to_extri_intri(pose_t.float(), (H, W))
    R_s, t_s = ext_s[..., :3, :3], ext_s[..., :3, 3]
    R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]

    S = R_s.shape[1]
    rot_terms, tdir_terms = [], []
    for i in range(S):
        for j in range(i + 1, S):
            Rr_s = R_s[:, i].transpose(-1, -2) @ R_s[:, j]
            Rr_t = R_t[:, i].transpose(-1, -2) @ R_t[:, j]
            e_p = ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean().clamp_min(1e-12)
            rot_terms.append(torch.nn.functional.huber_loss(
                e_p.sqrt(), torch.zeros_like(e_p), delta=delta))
            tr_s = R_s[:, i].transpose(-1, -2) @ (t_s[:, j] - t_s[:, i])[..., None]
            tr_t = R_t[:, i].transpose(-1, -2) @ (t_t[:, j] - t_t[:, i])[..., None]
            tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
            tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
            tdir_terms.append((1.0 - (tn_s * tn_t).sum(-1)).mean())
    rot_loss = torch.stack(rot_terms).mean()
    tdir_loss = torch.stack(tdir_terms).mean()
    return rot_loss + tdir_loss, {"rel_rot_cap": float(rot_loss), "rel_tdir": float(tdir_loss)}


def loss_triangle_pose(pose_s: torch.Tensor, pose_tN: torch.Tensor, n_anchors: int = 4):
    """Two-hop triangle closure: the student's direct relative pose i->j must
    match the teacher's composed i->e->j through extra-view anchor e (anchors
    enter teacher-side only). Gauge-free: relative rotation chordal +
    translation-direction 1-cos, Huber-capped per (i,j,e) path."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        H, W = 378, 504
        ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
        ext_t, _ = pose_encoding_to_extri_intri(pose_tN.detach().float(), (H, W))
        Rs, ts = ext_s[0, ..., :3, :3], ext_s[0, ..., :3, 3]      # [S,3,3],[S,3]
        Rt, tt = ext_t[0, ..., :3, :3], ext_t[0, ..., :3, 3]      # [N,3,3],[N,3]
        S = Rs.shape[0]
        N = Rt.shape[0]
        shared_slots = list(STUDENT_INDICES)
        extra_slots = [i for i in range(N) if i not in shared_slots]
        if not extra_slots:
            return torch.zeros((), device=Rs.device, requires_grad=True), {"tri_p": 0.0}
        step_a = max(1, len(extra_slots) // n_anchors)
        anchors = extra_slots[::step_a][:n_anchors]
        terms = []
        for i in range(S):
            for j in range(i + 1, S):
                R_ij_s = Rs[i].transpose(-1, -2) @ Rs[j]
                t_ij_s = Rs[i].transpose(-1, -2) @ (ts[j] - ts[i])
                ti, tj = shared_slots[i], shared_slots[j]
                for e in anchors:
                    # teacher two-hop i->e->j (teacher gauge, all from Rt/tt)
                    R_ie = Rt[ti].transpose(-1, -2) @ Rt[e]
                    t_ie = Rt[ti].transpose(-1, -2) @ (tt[e] - tt[ti])
                    R_ej = Rt[e].transpose(-1, -2) @ Rt[tj]
                    t_ej = Rt[e].transpose(-1, -2) @ (tt[tj] - tt[e])
                    R_ij_t = R_ej @ R_ie
                    t_ij_t = (R_ej @ t_ie.unsqueeze(-1)).squeeze(-1) + t_ej
                    e_rot = ((R_ij_s - R_ij_t) ** 2).sum().clamp_min(1e-12).sqrt()
                    e_rot = torch.nn.functional.huber_loss(
                        e_rot, torch.zeros_like(e_rot), delta=0.2)
                    tn_s = torch.nn.functional.normalize(t_ij_s, dim=-1, eps=1e-8)
                    tn_t = torch.nn.functional.normalize(t_ij_t, dim=-1, eps=1e-8)
                    e_t = (1.0 - (tn_s * tn_t).sum(-1)).mean()
                    terms.append(e_rot + e_t)
        loss = torch.stack(terms).mean()
        return loss, {"tri_p": float(loss)}


def loss_triangle_camtok(camera_head, teacher_cache, feats24_s, n_anchors: int = 4):
    """Feature-space triangle on camera tokens (L23, post token_norm): for each
    shared pair (i,j) and extra-frame anchor e, match the student's similarity
    triple (ci.cj, ci.ae, cj.ae) to the teacher's. Anchors = teacher camera
    tokens of extra views (teacher-side only)."""
    with torch.autocast(device_type="cuda", enabled=False):
        cs = camera_head.token_norm(feats24_s[23][:, :, 0, :].float())          # [1,S,C] grad
        ct_all = camera_head.token_norm(
            teacher_cache["feats"][23][0, :, 0, :].float()).detach()           # [N,C]
        ct = ct_all[list(STUDENT_INDICES)]                                      # [S,C]
        extra_slots = [i for i in range(ct_all.shape[0]) if i not in STUDENT_INDICES]
        if not extra_slots:
            return torch.zeros((), device=cs.device, requires_grad=True), {"tri_f": 0.0}
        step_a = max(1, len(extra_slots) // n_anchors)
        A = torch.nn.functional.normalize(
            ct_all[extra_slots[::step_a][:n_anchors]], dim=-1)                  # [A,C]
        cs_n = torch.nn.functional.normalize(cs[0], dim=-1)                     # [S,C]
        ct_n = torch.nn.functional.normalize(ct, dim=-1)
        S = cs_n.shape[0]
        terms = []
        for i in range(S):
            for j in range(i + 1, S):
                ss_ij = (cs_n[i] * cs_n[j]).sum(-1)
                tt_ij = (ct_n[i] * ct_n[j]).sum(-1)
                prof_s = torch.cat([(cs_n[i] * A).sum(-1), (cs_n[j] * A).sum(-1)])
                prof_t = torch.cat([(ct_n[i] * A).sum(-1), (ct_n[j] * A).sum(-1)])
                terms.append(torch.nn.functional.smooth_l1_loss(ss_ij, tt_ij, beta=0.1)
                             + torch.nn.functional.smooth_l1_loss(prof_s, prof_t, beta=0.1))
        loss = torch.stack(terms).mean()
        return loss, {"tri_f": float(loss)}


def loss_triangle_camtok_v2(camera_head, teacher_cache, feats24_s, n_anchors: int = 4,
                            temp: float = 0.07):
    """Richer feature-triangle: (a) softmax distribution over the extra-frame
    anchor bank (temperature temp), matched by KL(teacher || student) -
    captures anchor RANKING, scale-robust; (b) pairwise L2 distance between
    shared camera tokens, Huber - keeps norm/magnitude information that cosine
    discards. Anchors = teacher extra-view camera tokens (teacher-side only)."""
    with torch.autocast(device_type="cuda", enabled=False):
        cs = camera_head.token_norm(feats24_s[23][:, :, 0, :].float())          # [1,S,C] grad
        ct_all = camera_head.token_norm(
            teacher_cache["feats"][23][0, :, 0, :].float()).detach()           # [N,C]
        ct = ct_all[list(STUDENT_INDICES)]
        extra_slots = [i for i in range(ct_all.shape[0]) if i not in STUDENT_INDICES]
        if not extra_slots:
            return torch.zeros((), device=cs.device, requires_grad=True), {"tri_f2": 0.0}
        step_a = max(1, len(extra_slots) // n_anchors)
        A = torch.nn.functional.normalize(
            ct_all[extra_slots[::step_a][:n_anchors]], dim=-1)                  # [A,C]
        cs_n = torch.nn.functional.normalize(cs[0], dim=-1)                     # [S,C]
        ct_n = torch.nn.functional.normalize(ct, dim=-1)
        # (a) KL over anchor-bank distributions, per shared view
        logp_s = torch.nn.functional.log_softmax((cs_n @ A.T) / temp, dim=-1)   # [S,A]
        p_t = torch.nn.functional.softmax((ct_n @ A.T) / temp, dim=-1)
        kl = torch.nn.functional.kl_div(logp_s, p_t, reduction="batchmean")
        # (b) pairwise L2 distance matching (norm-aware)
        S = cs.shape[1]
        ds_ = torch.cdist(cs, cs).squeeze(0)                                    # [S,S]
        dt_ = torch.cdist(ct, ct).squeeze(0)
        iu = torch.triu_indices(S, S, offset=1, device=cs.device)
        dist = torch.nn.functional.smooth_l1_loss(ds_[iu[0], iu[1]], dt_[iu[0], iu[1]], beta=0.1)
        loss = kl + dist
        return loss, {"tri_f2_kl": float(kl), "tri_f2_dist": float(dist)}


def _robust_pdist(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Pairwise robust distance: mean per-channel Huber_8 (extreme channels
    capped) instead of L2. x: [1,S,C] -> [S,S]."""
    if x.dim() == 3:
        x = x[0]
    x = x.float()
    S = x.shape[0]
    dif = (x.unsqueeze(1) - x.unsqueeze(0)).abs()          # [S,S,C]
    hub = torch.where(dif <= delta, 0.5 * dif * dif, delta * (dif - 0.5 * delta))
    return hub.mean(-1)


def loss_triangle_camtok_v3(camera_head, teacher_cache, feats24_s, n_anchors: int = 4,
                            temp: float = 0.07, delta: float = 1.0):
    """TRIF2 with a robust metric: pairwise token distance = per-channel Huber
    mean (outlier-channel-proof), matched with Huber; anchor-bank KL profile
    unchanged."""
    with torch.autocast(device_type="cuda", enabled=False):
        cs = camera_head.token_norm(feats24_s[23][:, :, 0, :].float())
        ct_all = camera_head.token_norm(
            teacher_cache["feats"][23][0, :, 0, :].float()).detach()
        ct = ct_all[list(STUDENT_INDICES)]
        extra_slots = [i for i in range(ct_all.shape[0]) if i not in STUDENT_INDICES]
        if not extra_slots:
            return torch.zeros((), device=cs.device, requires_grad=True), {"tri_f3": 0.0}
        step_a = max(1, len(extra_slots) // n_anchors)
        A = torch.nn.functional.normalize(
            ct_all[extra_slots[::step_a][:n_anchors]], dim=-1)
        cs_n = torch.nn.functional.normalize(cs[0], dim=-1)
        ct_n = torch.nn.functional.normalize(ct, dim=-1)
        logp_s = torch.nn.functional.log_softmax((cs_n @ A.T) / temp, dim=-1)
        p_t = torch.nn.functional.softmax((ct_n @ A.T) / temp, dim=-1)
        kl = torch.nn.functional.kl_div(logp_s, p_t, reduction="batchmean")
        ds_ = _robust_pdist(cs, delta)
        dt_ = _robust_pdist(ct, delta)
        S = cs.shape[1]
        iu = torch.triu_indices(S, S, offset=1, device=cs.device)
        dist = torch.nn.functional.smooth_l1_loss(ds_[iu[0], iu[1]], dt_[iu[0], iu[1]], beta=0.1)
        loss = kl + dist
        return loss, {"tri_f3_kl": float(kl), "tri_f3_dist": float(dist)}


def loss_mc_consistency(depth_head, featsA, featsB):
    """Multi-context consistency: same student's post-LN patch features on the
    shared views, produced under a 4-view masked context vs the full teacher
    context (16v input to the STUDENT), must agree. Bridges the 4v->100v
    eval-time distribution gap. featsB is detached-free (same model, grad)."""
    total = 0.0
    for layer in TAP_LAYERS:
        ha = M.to_norm(depth_head, M.to_patch(featsA[layer].float()))
        hb = M.to_norm(depth_head, M.to_patch(featsB[layer][:, STUDENT_INDICES].float()))
        total = total + _huber_cos(ha, hb)
    v = total / len(TAP_LAYERS)
    return v, {"mc": float(v)}


def loss_xco_consistency(depth_head, featsA, featsB):
    """Co-student overlap consistency (teacher-free): one student, two
    overlapping 4-view contexts A=[0,2,4,6], B=[2,4,6,7]; the frames shared by
    both contexts (A[1:4] == B[0:3] == frames 2,4,6) must produce consistent
    post-LN patch features regardless of context. Attacks the <=8v bottleneck
    (cross-context pose/depth inconsistency) without pairwise similarity."""
    total = 0.0
    for layer in TAP_LAYERS:
        ha = M.to_norm(depth_head, M.to_patch(featsA[layer][:, 1:4].float()))
        hb = M.to_norm(depth_head, M.to_patch(featsB[layer][:, 0:3].float()))
        total = total + _huber_cos(ha, hb)
    v = total / len(TAP_LAYERS)
    return v, {"xco": float(v)}


EXTRA_INDICES = [i for i in range(8) if i not in STUDENT_INDICES]
XCO_B = [2, 4, 6, 7]


def mask_by_conf(images, conf4, patch_hw, mode, ratio=0.5):
    """Confidence-guided token masking: per view, mask the ratio fraction of
    patches with the LOWEST (mode='lo') or HIGHEST (mode='hi') teacher depth
    confidence. Returns (masked_images, patch_mask[1,S,P], 1=masked)."""
    conf = conf4.float()
    if conf.dim() == 5:
        conf = conf.squeeze(2)
    B, S, H, W = conf.shape
    ph, pw = patch_hw
    cfp = torch.nn.functional.avg_pool2d(
        conf.reshape(B * S, 1, H, W), kernel_size=(H // ph, W // pw)).reshape(B, S, ph * pw)
    k = max(1, int(round(ratio * ph * pw)))
    if mode == "lo":
        idx = cfp.argsort(dim=-1)[..., :k]
    else:
        idx = cfp.argsort(dim=-1, descending=True)[..., :k]
    mask = torch.zeros(B, S, ph * pw, device=images.device)
    mask.scatter_(-1, idx, 1.0)
    m = mask.reshape(B * S, ph, pw)
    m = m.repeat_interleave(H // ph, dim=1).repeat_interleave(W // pw, dim=2)
    out = images * (1.0 - m).reshape(B, S, 1, H, W)
    return out, mask


def loss_ss8m(depth_head, teacher_cache, feats8_s, patch_hw, patch_mask):
    """Self-supervised token completion on ALL 8 views: student sees the full
    8-view input with 50% image blocks zeroed; teacher features on all 8 views
    supervise ONLY the masked positions (uniform weight, post-LN Huber+2cos).
    Not the 8->4 paradigm: same information budget on both sides, student must
    reconstruct missing tokens from cross-view context."""
    w = patch_mask / patch_mask.mean().clamp_min(1e-8)
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_norm(depth_head, M.to_patch(feats8_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer].float()))
        total = total + _huber_cos(hs, ht, w)
    return total / len(TAP_LAYERS), {"mask_ratio": float(patch_mask.mean())}


def mask_image_blocks(images, ratio, patch_hw, gen):
    """Zero random 14x14 image blocks. Returns (masked_images, patch_mask[1,S,P])."""
    B, S, C, H, W = images.shape
    ph, pw = patch_hw
    block = (torch.rand(S, ph, pw, generator=gen, device=images.device) < ratio)
    m = block.repeat_interleave(H // ph, dim=1).repeat_interleave(W // pw, dim=2)
    out = images * (~m)[:, None].float()
    return out, block.reshape(1, S, ph * pw).float()


def loss_b5_maskdistill(depth_head, teacher_cache, feats24_s, patch_hw, patch_mask):
    """Masked distillation (iBOT/MGD mechanism): student sees block-masked
    images; the conf-gated B5 loss is computed ONLY on masked positions, so
    the LoRA-adapted attention must reconstruct the teacher's features there
    from cross-view context instead of copying local evidence.

    --v2_allpos: masked input unchanged, but the loss runs on ALL patch
    positions (w = teacher_patch_conf, no *patch_mask) and extra carries
    "allpos": 1.0. A None patch_mask already means all positions and is
    unaffected by the switch."""
    w_all = teacher_patch_conf(teacher_cache, patch_hw)
    mask_ratio = float(patch_mask.mean()) if patch_mask is not None else -1.0
    w = w_all if (_V2_ALLPOS or patch_mask is None) else w_all * patch_mask
    w = w / w.mean().clamp_min(1e-8)
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
        total = total + _huber_cos(hs, ht, w)
    # "feat" feeds the swanlab per-pair feature curves; "mask_ratio" always
    # reports the INPUT mask (still ~0.5 under --v2_allpos)
    extra = {"mask_ratio": mask_ratio,
             "feat": float(total / len(TAP_LAYERS))}
    if _V2_ALLPOS:
        extra["allpos"] = 1.0
    return total / len(TAP_LAYERS), extra


def teacher_patch_conf(teacher_cache, patch_hw) -> torch.Tensor:
    """Teacher 8->4 depth_conf [1,4,1,H,W] -> per-patch weight [1,4,P], mean 1."""
    conf = teacher_cache["conf4"].float()
    if conf.dim() == 5:
        conf = conf.squeeze(2)
    B, S, H, W = conf.shape
    ph, pw = patch_hw
    confp = torch.nn.functional.avg_pool2d(
        conf.reshape(B * S, 1, H, W), kernel_size=(H // ph, W // pw))
    confp = confp.reshape(B, S, ph * pw)
    w = confp / confp.mean().clamp_min(1e-8)
    return w.detach()


def loss_b5_confperm(depth_head, teacher_cache, feats24_s, patch_hw,
                     layers=None, layer_w=None) -> torch.Tensor:
    """Control arm: B5 with teacher-conf weights permuted across patches.
    Same marginal weight distribution, destroyed patch correspondence. If this
    ties with B5_conf, the gate's value is reweighting, not selection."""
    w = teacher_patch_conf(teacher_cache, patch_hw)
    g = torch.Generator().manual_seed(9871)
    idx = torch.randperm(w.shape[-1], generator=g).to(w.device)
    w = w[:, :, idx].detach()
    layers = layers or TAP_LAYERS
    layer_w = layer_w or {}
    total, wsum = 0.0, 0.0
    for layer in layers:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
        wl = layer_w.get(layer, 1.0)
        total = total + wl * _huber_cos(hs, ht, w)
        wsum += wl
    return total / wsum, {"conf_w_max": float(w.max()), "conf_w_min": float(w.min())}


def student_patch_conf(preds, patch_hw) -> torch.Tensor:
    """Student depth_conf (detached) -> per-patch weight [1,4,P], mean 1."""
    conf = preds["conf"].float()
    if conf.dim() == 5:
        conf = conf.squeeze(2)
    B, S, H, W = conf.shape
    ph, pw = patch_hw
    confp = torch.nn.functional.avg_pool2d(
        conf.reshape(B * S, 1, H, W), kernel_size=(H // ph, W // pw))
    confp = confp.reshape(B, S, ph * pw)
    w = confp / confp.mean().clamp_min(1e-8)
    return w.detach()


def _b5_form_with_w(depth_head, teacher_cache, feats24_s, patch_hw, w) -> torch.Tensor:
    """B5 form (post-norm Huber+2cos, all 4 tap layers) with a caller-supplied
    per-patch weight (mean-normalized, detached)."""
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
        total = total + _huber_cos(hs, ht, w)
    return total / len(TAP_LAYERS)


def loss_b5_conf(depth_head, teacher_cache, feats24_s, patch_hw,
                 layers=None, layer_w=None) -> torch.Tensor:
    """B5: B1 (norm-space patch Huber+cos) gated by teacher depth confidence.
    layers: subset of TAP_LAYERS (default all four); layer_w: optional
    per-layer weight dict (losses stay normalized by sum of weights)."""
    w = teacher_patch_conf(teacher_cache, patch_hw)
    layers = layers or TAP_LAYERS
    layer_w = layer_w or {}
    total, wsum = 0.0, 0.0
    for layer in layers:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
        wl = layer_w.get(layer, 1.0)
        total = total + wl * _huber_cos(hs, ht, w)
        wsum += wl
    # "feat" feeds the swanlab per-pair feature curves
    return total / wsum, {"conf_w_max": float(w.max()), "conf_w_min": float(w.min()),
                          "feat": float(total / wsum)}


def loss_b_patchform(depth_head, teacher_cache, feats24_s, patch_hw, space: str,
                     cam_token: bool):
    """B-arms: deployed Huber+2cos form on PATCH tokens in raw|norm space,
    optional + camera-token MSE at layer 23."""
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_patch(feats24_s[layer].float())
        ht = M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float())
        if space == "norm":
            hs = M.to_norm(depth_head, hs)
            ht = M.to_norm(depth_head, ht)
        total = total + _huber_cos(hs, ht)
    loss = total / len(TAP_LAYERS)
    extra = {}
    if cam_token:
        cs = feats24_s[23][:, :, 0, :].float()
        ct = teacher_cache["feats"][23][:, STUDENT_INDICES, 0, :].float()
        cam = torch.mean((cs - ct) ** 2)
        loss = loss + cam
        extra["camtok"] = float(cam)
    return loss, extra


def loss_pose_rel(pose_s: torch.Tensor, pose_t: torch.Tensor) -> torch.Tensor:
    """Relative-pose loss, CORRECTED w2c construction (2026-09-18):
    T_{i<-j} = E_i @ inv(E_j), NOT inv(E_i) @ E_j.
    R_rel = R_i @ R_j^T,  t_rel = t_i - R_i @ R_j^T @ t_j.
    Loss form unchanged: rotation chordal + translation-direction 1-cos.
    pose_*: [1,S,9] pose encodings."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    H, W = 378, 504
    ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
    ext_t, _ = pose_encoding_to_extri_intri(pose_t.float(), (H, W))
    R_s, t_s = ext_s[..., :3, :3], ext_s[..., :3, 3]
    R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]

    S = R_s.shape[1]
    rot_loss, tdir_loss, npairs = 0.0, 0.0, 0
    for i in range(S):
        for j in range(i + 1, S):
            Rr_s = R_s[:, i] @ R_s[:, j].transpose(-1, -2)
            Rr_t = R_t[:, i] @ R_t[:, j].transpose(-1, -2)
            # SHAPE FIX 2026-09-18: t[:,i] is [B,3]; t[:,j][...,None] is [B,3,1].
            # [B,3] - [B,3,1] broadcasts to [B,3,3] (a MATRIX) — wrong!
            # Must keep both as column vectors [B,3,1]:
            tr_s = t_s[:, i].unsqueeze(-1) - Rr_s @ t_s[:, j].unsqueeze(-1)
            tr_t = t_t[:, i].unsqueeze(-1) - Rr_t @ t_t[:, j].unsqueeze(-1)
            assert tr_s.shape == (t_s.shape[0], 3, 1), \
                f"tr_s shape {tr_s.shape} != ({t_s.shape[0]}, 3, 1)"
            assert tr_t.shape == tr_s.shape
            rot_loss = rot_loss + ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
            tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
            tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
            tdir_loss = tdir_loss + (1.0 - (tn_s * tn_t).sum(-1)).mean()
            npairs += 1
    rot_loss = rot_loss / npairs
    tdir_loss = tdir_loss / npairs
    return rot_loss + tdir_loss, {"rel_rot": float(rot_loss), "rel_tdir": float(tdir_loss)}


def _kabsch_rt(P_s: torch.Tensor, P_t: torch.Tensor):
    """Rigid (R,t) aligning point set P_s onto P_t. P: [N,3]. Detached caller-side."""
    cs, ct = P_s.mean(0), P_t.mean(0)
    X, Y = P_s - cs, P_t - ct
    U, _, Vh = torch.linalg.svd(X.T @ Y)
    d = torch.sign(torch.linalg.det(Vh.mT @ U.mT))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vh.mT @ D @ U.mT
    return R, ct - R @ cs


def loss_pose_rel_x(pose_s: torch.Tensor, pose_t8: torch.Tensor) -> torch.Tensor:
    """Cross-gauge relative-pose loss with teacher-extra views as anchors.

    Student abs poses and teacher abs poses live in different Sim(3) gauges, so
    the student's trajectory is first rigidly aligned to the teacher's shared-4
    cameras (Kabsch on the 4 camera centers, alignment detached). Loss =
    6 shared-shared rel-pose pairs (gauge-invariant, as loss_pose_rel)
    + 16 shared->extra rel-pose pairs in the teacher gauge (the extra 4 frames'
    information entering ONLY through the pose path).
    pose_s: [1,4,9] (grad), pose_t8: [1,8,9] (detached teacher)."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        H, W = 378, 504
        ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
        ext_t, _ = pose_encoding_to_extri_intri(pose_t8.detach().float(), (H, W))
        R_s, t_s = ext_s[0, :, :3, :3].float(), ext_s[0, :, :3, 3].float()   # [4,3,3],[4,3]
        R_t, t_t = ext_t[0, :, :3, :3].float(), ext_t[0, :, :3, 3].float()   # [8,3,3],[8,3]

        with torch.no_grad():
            Ra, ta = _kabsch_rt(t_s, t_t[STUDENT_INDICES])
        R_sa, t_sa = Ra @ R_s, (Ra @ t_s.T).T + ta                # aligned student
        R_ts, t_ts = R_t[STUDENT_INDICES], t_t[STUDENT_INDICES]   # teacher shared
        R_te, t_te = R_t[EXTRA_INDICES], t_t[EXTRA_INDICES]       # teacher extra

        def pair_terms(Ra_, ta_, Rb_, tb_, Rc_, tc_, Rd_, td_):
            Rr_s = Ra_.transpose(-1, -2) @ Rb_
            Rr_t = Rc_.transpose(-1, -2) @ Rd_
            tr_s = Ra_.transpose(-1, -2) @ (tb_ - ta_)[..., None]
            tr_t = Rc_.transpose(-1, -2) @ (td_ - tc_)[..., None]
            rot = ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
            tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
            tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
            return rot + (1.0 - (tn_s * tn_t).sum(-1)).mean()

        ss_loss, npairs = 0.0, 0
        for i in range(4):
            for j in range(i + 1, 4):
                ss_loss = ss_loss + pair_terms(R_sa[i], t_sa[i], R_sa[j], t_sa[j],
                                               R_ts[i], t_ts[i], R_ts[j], t_ts[j])
                npairs += 1
        sx_loss, nx = 0.0, 0
        for i in range(4):
            for j in range(4):
                sx_loss = sx_loss + pair_terms(R_sa[i], t_sa[i], R_te[j], t_te[j],
                                               R_ts[i], t_ts[i], R_te[j], t_te[j])
                nx += 1
        ss_loss, sx_loss = ss_loss / npairs, sx_loss / nx
        return ss_loss + sx_loss, {"xrel_ss": float(ss_loss), "xrel_sx": float(sx_loss)}


def loss_pose_rel_x2(pose_s: torch.Tensor, pose_t8: torch.Tensor, w_sx: float = 1.0):
    """Shared<->extra pose-anchor loss, NO gauge alignment.

    Verified empirically (18 pairs): teacher 8v and student 4v absolute poses
    are already frame-consistent (first-camera anchored, view0 rot err = 0.0,
    mean abs rot err 1.0 deg; Kabsch alignment on 4 centers made it 5.8 deg
    WORSE). So decode both directly and constrain
      6 shared-shared rel pairs (as loss_pose_rel)
      + w_sx * 16 shared(student, grad) <-> extra(teacher, detached) rel pairs.
    Rotation chordal + translation DIRECTION cosine (scale-free, absorbs the
    ~4% per-pair scale difference)."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.autocast(device_type="cuda", enabled=False):
        H, W = 378, 504
        ext_s, _ = pose_encoding_to_extri_intri(pose_s.float(), (H, W))
        ext_t, _ = pose_encoding_to_extri_intri(pose_t8.detach().float(), (H, W))
        R_s, t_s = ext_s[0, :, :3, :3].float(), ext_s[0, :, :3, 3].float()
        R_t, t_t = ext_t[0, :, :3, :3].float(), ext_t[0, :, :3, 3].float()
        R_ts, t_ts = R_t[STUDENT_INDICES], t_t[STUDENT_INDICES]
        R_te, t_te = R_t[EXTRA_INDICES], t_t[EXTRA_INDICES]

        def pair_terms(Ra_, ta_, Rb_, tb_, Rc_, tc_, Rd_, td_):
            Rr_s = Ra_.transpose(-1, -2) @ Rb_
            Rr_t = Rc_.transpose(-1, -2) @ Rd_
            tr_s = Ra_.transpose(-1, -2) @ (tb_ - ta_)[..., None]
            tr_t = Rc_.transpose(-1, -2) @ (td_ - tc_)[..., None]
            rot = ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
            tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
            tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
            return rot + (1.0 - (tn_s * tn_t).sum(-1)).mean()

        ss_loss, np_ = 0.0, 0
        for i in range(4):
            for j in range(i + 1, 4):
                ss_loss = ss_loss + pair_terms(R_s[i], t_s[i], R_s[j], t_s[j],
                                               R_ts[i], t_ts[i], R_ts[j], t_ts[j])
                np_ += 1
        sx_loss, nx = 0.0, 0
        for i in range(4):
            for j in range(4):
                sx_loss = sx_loss + pair_terms(R_s[i], t_s[i], R_te[j], t_te[j],
                                               R_ts[i], t_ts[i], R_te[j], t_te[j])
                nx += 1
        ss_loss, sx_loss = ss_loss / np_, sx_loss / nx
        return ss_loss + w_sx * sx_loss, {"x2_ss": float(ss_loss), "x2_sx": float(sx_loss)}


def loss_camrel(depth_head, camera_head, teacher_cache, feats24_s, n_p=128, tau=1.0):
    """Camera-token affinity-profile distillation (post-LN).

    For each shared view's camera token, match the cosine-affinity profile to
    a stratified anchor bank of teacher patch tokens: 128 shared-view patches
    + 128 EXTRA-view patches. The extra-view anchors inject the extra 4 frames'
    information into the camera path in feature space (the xpose idea, but on
    camera tokens instead of decoded poses). Camera tokens are read through
    camera_head.token_norm, matching the head's actual input."""
    with torch.autocast(device_type="cuda", enabled=False):
        ht8 = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][23].float()))
        C = ht8.shape[-1]
        A_sh = ht8[:, STUDENT_INDICES].reshape(1, -1, C)
        A_ex = ht8[:, EXTRA_INDICES].reshape(1, -1, C)
        idx_sh = torch.randperm(A_sh.shape[1], device=A_sh.device)[:n_p]
        idx_ex = torch.randperm(A_ex.shape[1], device=A_ex.device)[:n_p]
        A = torch.nn.functional.normalize(
            torch.cat([A_sh[:, idx_sh], A_ex[:, idx_ex]], 1), dim=-1).detach()  # [1,2n_p,C]
        cs = camera_head.token_norm(feats24_s[23][:, :, 0, :].float())          # [1,4,C] grad
        ct = camera_head.token_norm(
            teacher_cache["feats"][23][:, STUDENT_INDICES, 0, :].float()).detach()
        cs_n = torch.nn.functional.normalize(cs, dim=-1)
        ct_n = torch.nn.functional.normalize(ct, dim=-1)
        prof_s = (cs_n.unsqueeze(2) * A).sum(-1)                               # [1,4,2n_p]
        prof_t = (ct_n.unsqueeze(2) * A).sum(-1)
        loss = ((prof_s - prof_t) ** 2).mean() / tau
        return loss, {"camrel": float(loss)}


def loss_camtok_hc(camera_head, teacher_cache, feats24_s, layers=None):
    """Camera-token loss in the SAME form as the patch loss (Huber beta=1 +
    2*(1-cos)), in the camera head's own readout space (token_norm). B2/B4
    used plain MSE (outlier-amplifying); this tests whether the earlier
    camera-token failures were the loss form rather than the token itself.
    layers: which tap layers to supervise (default [23]; the successful
    alpha-interpolation swapped ALL layers, so multi-layer may carry more)."""
    with torch.autocast(device_type="cuda", enabled=False):
        total = 0.0
        for layer in (layers or [23]):
            cs = camera_head.token_norm(feats24_s[layer][:, :, 0, :].float())
            ct = camera_head.token_norm(
                teacher_cache["feats"][layer][:, STUDENT_INDICES, 0, :].float()).detach()
            huber = torch.nn.functional.smooth_l1_loss(cs, ct, beta=1.0)
            cos = torch.nn.functional.cosine_similarity(cs, ct, dim=-1).mean()
            total = total + huber + 2.0 * (1.0 - cos)
        layers_n = len(layers or [23])
        return total / layers_n, {"camtok_hc": float(total / layers_n)}


def perpatch_logres2(depth, gt4, patch_hw):
    """[4,H,W] depth vs GT -> per-patch centered squared log residual [4,P].
    Pair-wide scale correction, mirroring E_depth."""
    ph, pw = patch_hw
    depth = np.asarray(depth, dtype=np.float64)
    gt4 = np.asarray(gt4, dtype=np.float64)
    r = np.log(np.clip(depth, 1e-6, None)) - np.log(np.clip(gt4, 1e-6, None))
    valid = np.isfinite(gt4) & (gt4 > 0)
    r = np.where(valid, r, 0.0)
    c = r.sum() / max(valid.sum(), 1)
    r2 = np.where(valid, (r - c) ** 2, 0.0)
    S, H, W = r2.shape
    return r2.reshape(S, ph, H // ph, pw, W // pw).mean(axis=(2, 4)).reshape(S, ph * pw)


def loss_g1_oracle(depth_head, teacher_cache, feats24_s, patch_hw):
    """G1: B1 + ORACLE per-patch gate (w=1 where teacher locally better than the
    frozen student by GT depth, else 0.1). ORACLE ONLY - diagnostic ceiling for
    gating, not usable as a paper method (uses GT)."""
    w = teacher_cache["oracle_w"]
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
        total = total + _huber_cos(hs, ht, w)
    return total / len(TAP_LAYERS), {"gate_mean": float(w.mean())}


def loss_g2_interp(depth_head, teacher_cache, feats24_s, patch_hw):
    """G2: B1 but the target is the midpoint H_base + 0.5*(H_teacher - H_base)
    (static, computed from the frozen student) - half-way teacher, less noise."""
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, teacher_cache["interp_target"][layer].float())
        total = total + _huber_cos(hs, ht)
    return total / len(TAP_LAYERS), {}


def loss_g4_l23only(depth_head, teacher_cache, feats24_s, patch_hw):
    """G4: B5 (conf-gated) but supervise ONLY layer 23 (gradient energy is
    ~100x concentrated there; also the memory-cheapest feature loss)."""
    w = teacher_patch_conf(teacher_cache, patch_hw)
    layer = 23
    hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
    ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
    return _huber_cos(hs, ht, w), {}


# ---------------------------------------------------------------- Phase D

def loss_d1_relfeat(depth_head, teacher_cache, feats24_s, patch_hw, n_samples=256):
    """D1: relational feature distillation on layer 23 - match the pairwise
    cosine-similarity MATRIX between sampled patch tokens (per view), instead
    of absolute features. 'Relative' supervision, no absolute noise import."""
    hs = M.to_norm(depth_head, M.to_patch(feats24_s[23].float()))   # [1,4,P,C]
    ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][23][:, STUDENT_INDICES].float()))
    B, S, P, C = hs.shape
    idx = torch.randperm(P, device=hs.device)[:n_samples]
    hs = torch.nn.functional.normalize(hs[:, :, idx, :], dim=-1)
    ht = torch.nn.functional.normalize(ht[:, :, idx, :], dim=-1)
    sim_s = hs @ hs.transpose(-1, -2)   # [1,4,Q,Q]
    sim_t = ht @ ht.transpose(-1, -2)
    return torch.mean((sim_s - sim_t.detach()) ** 2), {}


def loss_d2_layermean(depth_head, teacher_cache, feats24_s, patch_hw):
    """D2: average the four tap-layer tokens AFTER per-layer LayerNorm (scale-
    aligned averaging), one Huber+cos on the averaged channel."""
    hs_l, ht_l = [], []
    for layer in TAP_LAYERS:
        hs_l.append(M.to_norm(depth_head, M.to_patch(feats24_s[layer].float())))
        ht_l.append(M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float())))
    return _huber_cos(torch.stack(hs_l).mean(0), torch.stack(ht_l).mean(0)), {}


def loss_d3_spatialpool(depth_head, teacher_cache, feats24_s, patch_hw, k=3):
    """D3: kxk spatial mean-pool of patch tokens (norm space) before matching -
    local aggregation denoises the target."""
    ph, pw = patch_hw
    w = teacher_patch_conf(teacher_cache, patch_hw)  # [1,4,P] on full grid
    B, S, P = w.shape
    wmap = w.reshape(B * S, 1, ph, pw)
    wmap = torch.nn.functional.avg_pool2d(wmap, k, stride=k)  # coarse grid
    wmap = (wmap / wmap.mean().clamp_min(1e-8)).reshape(B, S, -1)

    def pool(h):  # [1,4,P,C] -> [1,4,Pc,C]
        Bs, Ss, Pp, C = h.shape
        hm = h.reshape(Bs * Ss, ph, pw, C).permute(0, 3, 1, 2)
        hm = torch.nn.functional.avg_pool2d(hm, k, stride=k)
        return hm.reshape(Bs, Ss, C, -1).permute(0, 1, 3, 2)

    total = 0.0
    for layer in [17, 23]:
        hs = pool(M.to_norm(depth_head, M.to_patch(feats24_s[layer].float())))
        ht = pool(M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float())))
        total = total + _huber_cos(hs, ht, wmap)
    return total / 2, {}


def loss_d4_depthgrad(teacher_cache, preds):
    """D4: A3 (scale-invariant log-depth) + spatial-gradient matching on
    log-depth (x/y finite differences) - sharpens local geometry for recon."""
    base_loss, _ = loss_a3_outdepth(teacher_cache, preds)
    d_s = torch.log(preds["depth"].squeeze(0).squeeze(-1).float().clamp_min(1e-6))
    d_t = torch.log(teacher_cache["depth4"].squeeze(0).squeeze(-1).float().clamp_min(1e-6))
    valid = torch.isfinite(d_t)
    gx_s = d_s[:, :, 1:] - d_s[:, :, :-1]
    gx_t = d_t[:, :, 1:] - d_t[:, :, :-1]
    gy_s = d_s[:, 1:, :] - d_s[:, :-1, :]
    gy_t = d_t[:, 1:, :] - d_t[:, :-1, :]
    vx = valid[:, :, 1:] & valid[:, :, :-1]
    vy = valid[:, 1:, :] & valid[:, :-1, :]
    gl = (((gx_s - gx_t) ** 2) * vx).sum() / vx.sum().clamp_min(1) + \
         (((gy_s - gy_t) ** 2) * vy).sum() / vy.sum().clamp_min(1)
    return base_loss + gl, {"depth_grad": float(gl)}


def loss_e2_outdepth_se(teacher_cache, preds, gamma=1.0, alpha=0.2):
    """E2: SelfEvo-style output-depth loss, extending loss_a3_outdepth.
    (1) pseudo-label pruning: pixels below the pair-wise 5% teacher-conf
        quantile are invalid; (2) aleatoric regression
        gamma*conf_s*|dlogd| - alpha*log(conf_s) with conf_s = student
        depth_conf clamped to [1e-3, 1e3]; (3) per-pixel |dlogd| 98% quantile
        outlier cut before the mean; (4) multi-scale log-depth gradient
        matching (finite differences as in loss_d4_depthgrad, scales {1,2}).
    NOTE: |dlogd| is NOT scale-centered (SelfEvo form), unlike A3."""
    depth = preds["depth"].squeeze(0).squeeze(-1).float()          # [4,H,W]
    tdepth = teacher_cache["depth4"].squeeze(0).squeeze(-1).float()
    conf_s = preds["conf"].float().reshape(depth.shape).clamp(1e-3, 1e3)
    conf_t = teacher_cache["conf4"].float().reshape(depth.shape)

    log_s = torch.log(depth.clamp_min(1e-6))
    log_t = torch.log(tdepth.clamp_min(1e-6))
    valid = torch.isfinite(tdepth) & (tdepth > 0) & torch.isfinite(conf_t)
    n0 = int(valid.sum())
    if n0 > 0:
        thr = torch.quantile(conf_t[valid], 0.05)                # (1) pair-wise prune
        valid = valid & (conf_t >= thr)
        n0 = int(valid.sum())
    r = (log_s - log_t).abs()
    r = torch.where(valid, r, torch.zeros_like(r))
    if n0 > 0:
        q98 = torch.quantile(r.detach()[valid], 0.98)            # (3) outlier cut
        keep = valid & (r.detach() <= q98)
    else:
        keep = valid
    n = keep.sum().clamp_min(1)
    ale = gamma * conf_s * r - alpha * torch.log(conf_s)         # (2) aleatoric
    aleatoric = (ale * keep).sum() / n

    gl = 0.0                                                     # (4) grad match
    for scale in (1, 2):
        if scale == 1:
            ls, lt, v = log_s, log_t, keep
        else:
            ls = torch.nn.functional.avg_pool2d(log_s.unsqueeze(1), 2).squeeze(1)
            lt = torch.nn.functional.avg_pool2d(log_t.unsqueeze(1), 2).squeeze(1)
            v = torch.nn.functional.avg_pool2d(
                keep.float().unsqueeze(1), 2).squeeze(1) >= 0.999
        ls = torch.nan_to_num(ls, nan=0.0)
        lt = torch.nan_to_num(lt, nan=0.0)
        gx_s = ls[:, :, 1:] - ls[:, :, :-1]
        gx_t = lt[:, :, 1:] - lt[:, :, :-1]
        gy_s = ls[:, 1:, :] - ls[:, :-1, :]
        gy_t = lt[:, 1:, :] - lt[:, :-1, :]
        vx = v[:, :, 1:] & v[:, :, :-1]
        vy = v[:, 1:, :] & v[:, :-1, :]
        glx = torch.where(vx, (gx_s - gx_t) ** 2, torch.zeros_like(gx_s))
        gly = torch.where(vy, (gy_s - gy_t) ** 2, torch.zeros_like(gy_s))
        gl = gl + glx.sum() / vx.sum().clamp_min(1) + gly.sum() / vy.sum().clamp_min(1)
    loss = aleatoric + gl
    return loss, {"e2_aleatoric": float(aleatoric), "e2_grad": float(gl),
                  "e2_keep_frac": float(keep.sum() / max(n0, 1)) if n0 else 0.0,
                  "e2_conf_mean": float((conf_s * keep).sum() / n)}


def loss_c4_selfmix(depth_head, teacher_cache, feats24_s, patch_hw, mix=0.5):
    """C4: self-mix target = (1-mix)*teacher + mix*detach(student), conf-gated
    norm space (B5-style). Probes EMA-style drift/error-accumulation cheaply:
    if this collapses or drifts, pure self-EMA teachers are unsafe."""
    w = teacher_patch_conf(teacher_cache, patch_hw)
    total = 0.0
    for layer in TAP_LAYERS:
        hs = M.to_norm(depth_head, M.to_patch(feats24_s[layer].float()))
        ht = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
        target = ((1.0 - mix) * ht + mix * hs.detach()).detach()
        total = total + _huber_cos(hs, target, w)
    return total / len(TAP_LAYERS), {}


# ---------------------------------------------------------------- Phase R
# Repaired cross-frame (triangle) distillation: post-norm, patch-only, conf
# weighted, top-k anchors only, no KL. All run on layer 23 in true fp32.

EXTRA_INDICES = [i for i in range(8) if i not in STUDENT_INDICES]  # [1,3,5,7]


def _l23_norm_patches(depth_head, teacher_cache, feats24_s):
    """Layer-23 post-norm patch tokens. Returns (hs [1,4,P,C] with grad,
    ht8 [1,8,P,C] detached teacher)."""
    hs = M.to_norm(depth_head, M.to_patch(feats24_s[23].float()))
    ht8 = M.to_norm(depth_head, M.to_patch(teacher_cache["feats"][23].float())).detach()
    return hs, ht8


def _qk_match(q_s, q_t, anchors, wq, m, tau):
    """QK profile match for one view. q_s [1,Q,C] (grad), q_t [1,Q,C] and
    anchors [1,A,C] (detached). Top-m anchors per query chosen by the TEACHER
    cosine; per-query MSE(student cos, teacher cos)/tau, conf-weighted mean."""
    B, Q, C = q_s.shape
    q_s = torch.nn.functional.normalize(q_s, dim=-1)
    q_t = torch.nn.functional.normalize(q_t, dim=-1)
    anchors = torch.nn.functional.normalize(anchors, dim=-1)
    sim_t = q_t @ anchors.transpose(-1, -2)                      # [1,Q,A]
    top_idx = sim_t.topk(m, dim=-1).indices                      # [1,Q,m]
    a_sel = torch.gather(
        anchors.unsqueeze(1).expand(B, Q, anchors.shape[1], C), 2,
        top_idx.unsqueeze(-1).expand(B, Q, m, C))                # [1,Q,m,C]
    s_t = (q_t.unsqueeze(2) * a_sel).sum(-1)                     # [1,Q,m]
    s_s = (q_s.unsqueeze(2) * a_sel).sum(-1)
    lq = ((s_s - s_t) ** 2).mean(dim=-1) / tau                   # [1,Q]
    return (lq * wq).sum() / wq.sum().clamp_min(1e-8)


def loss_r1_qk(depth_head, teacher_cache, feats24_s, patch_hw, n_q=256, m=16, tau=0.5):
    """R1: QK similarity matching. Anchors = post-norm patch tokens of the
    teacher-extra views (all non-shared views; = [1,3,5,7] at ctx8, detached).
    Per shared view, n_q query patches are uniformly sampled; student
    query->anchor cosine profile is matched to the teacher's on the
    teacher-chosen top-m anchors (MSE/tau, conf-weighted)."""
    with torch.autocast(device_type="cuda", enabled=False):
        w = teacher_patch_conf(teacher_cache, patch_hw)          # [1,4,P]
        hs, ht8 = _l23_norm_patches(depth_head, teacher_cache, feats24_s)
        ht = ht8[:, STUDENT_INDICES]                             # [1,4,P,C]
        extra_idx = [i for i in range(ht8.shape[1]) if i not in STUDENT_INDICES]
        anchors = ht8[:, extra_idx].reshape(1, -1, ht8.shape[-1])  # [1,EP,C]
        B, S, P, C = hs.shape
        total = 0.0
        for v in range(S):
            idx = torch.randperm(P, device=hs.device)[:n_q]
            total = total + _qk_match(hs[:, v, idx, :], ht[:, v, idx, :],
                                      anchors, w[:, v, idx], m, tau)
        loss = total / S
        return loss, {"r1_qk": float(loss)}


def loss_r3_shared(depth_head, teacher_cache, feats24_s, patch_hw, n_q=256, m=16, tau=0.5):
    """R3: same as R1 but anchors = the OTHER 3 shared views' teacher patch
    tokens (query's own view excluded)."""
    with torch.autocast(device_type="cuda", enabled=False):
        w = teacher_patch_conf(teacher_cache, patch_hw)          # [1,4,P]
        hs, ht8 = _l23_norm_patches(depth_head, teacher_cache, feats24_s)
        ht = ht8[:, STUDENT_INDICES]                             # [1,4,P,C]
        B, S, P, C = hs.shape
        total = 0.0
        for v in range(S):
            idx = torch.randperm(P, device=hs.device)[:n_q]
            others = [u for u in range(S) if u != v]
            anchors = ht[:, others].reshape(1, -1, C)            # [1,3P,C]
            total = total + _qk_match(hs[:, v, idx, :], ht[:, v, idx, :],
                                      anchors, w[:, v, idx], m, tau)
        loss = total / S
        return loss, {"r3_shared": float(loss)}


def _tri_chunk(rs, ss, rt, at, st, a1_t, a2_t, a3_t, wc):
    """Student-side three-angle Huber sums for one ref/shared chunk (deployed
    angle form: cosine of difference vectors, Huber delta=1), conf-weighted.
    rs/ss have grad; rt/at/st/a*_t/wc are detached constants. Broadcast shapes:
    rs/rt [1,Rc,1,1,C], ss/st [1,1,Sc,1,C], at [1,Rc,1,K,C], a*_t [1,Rc,Sc,K]."""
    def cos(a, b):
        return (torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
                * torch.nn.functional.normalize(b, dim=-1, eps=1e-8)).sum(-1)

    a1_s = cos(ss - rs, at - rs)
    a2_s = cos(rs - at, ss - at)
    a3_s = cos(rs - ss, at - ss)
    l1 = (torch.nn.functional.huber_loss(a1_s, a1_t, reduction="none", delta=1.0) * wc).sum()
    l2 = (torch.nn.functional.huber_loss(a2_s, a2_t, reduction="none", delta=1.0) * wc).sum()
    l3 = (torch.nn.functional.huber_loss(a3_s, a3_t, reduction="none", delta=1.0) * wc).sum()
    return l1, l2, l3


def loss_r2_trifix(depth_head, teacher_cache, feats24_s, patch_hw, topk=4,
                   num_ref=128, num_shared=128, chunk=32):
    """R2: minimal fix of the deployed triangle (CF-angle) loss, rewritten on
    post-norm patch tokens: anchors = teacher-extra views' patches, top-k only
    (no least-similar half), NO CF-distance term, ref frame = shared frame 0,
    each ref token's terms weighted by its teacher conf. Chunk computations are
    checkpointed so no large [R,S,K,C] intermediates are cached."""
    import torch.utils.checkpoint as cp

    with torch.autocast(device_type="cuda", enabled=False):
        w = teacher_patch_conf(teacher_cache, patch_hw)          # [1,4,P]
        hs, ht8 = _l23_norm_patches(depth_head, teacher_cache, feats24_s)
        ht = ht8[:, STUDENT_INDICES]                             # [1,4,P,C]
        B, S, P, C = hs.shape
        num_ref = min(num_ref, P)
        num_shared = min(num_shared, P)
        ref_perm = torch.randperm(P, device=hs.device)[:num_ref]
        shared_perm = torch.randperm(P, device=hs.device)[:num_shared]

        ref_s = hs[:, 0, ref_perm, :]                            # [1,R,C] grad
        ref_t = ht[:, 0, ref_perm, :]                            # detached
        extra_idx = [i for i in range(ht8.shape[1]) if i not in STUDENT_INDICES]
        extra = ht8[:, extra_idx].reshape(1, -1, C)              # [1,EP,C]

        with torch.no_grad():
            sim = torch.nn.functional.normalize(ref_t, dim=-1) @ \
                torch.nn.functional.normalize(extra, dim=-1).transpose(-1, -2)
            top_idx = sim.topk(topk, dim=-1).indices             # [1,R,K]
            a_sel = torch.gather(
                extra.unsqueeze(1).expand(B, num_ref, extra.shape[1], C), 2,
                top_idx.unsqueeze(-1).expand(B, num_ref, topk, C))   # [1,R,K,C]

            def cos(a, b):
                return (torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
                        * torch.nn.functional.normalize(b, dim=-1, eps=1e-8)).sum(-1)

        w_ref = w[:, 0, ref_perm]                                # [1,R]
        sums = [0.0, 0.0, 0.0]
        wsum = 0.0
        for v in range(1, S):                                    # shared views 1..3
            sh_s = hs[:, v, shared_perm, :]
            sh_t = ht[:, v, shared_perm, :]
            for r0 in range(0, num_ref, chunk):
                r1 = min(r0 + chunk, num_ref)
                rs = ref_s[:, r0:r1, :].unsqueeze(2).unsqueeze(3)
                rt = ref_t[:, r0:r1, :].unsqueeze(2).unsqueeze(3)
                at = a_sel[:, r0:r1, :, :].unsqueeze(2)
                wc = w_ref[:, r0:r1].reshape(1, -1, 1, 1)
                for s0 in range(0, num_shared, chunk):
                    s1 = min(s0 + chunk, num_shared)
                    ss = sh_s[:, s0:s1, :].unsqueeze(1).unsqueeze(3)
                    st = sh_t[:, s0:s1, :].unsqueeze(1).unsqueeze(3)
                    with torch.no_grad():
                        a1_t = cos(st - rt, at - rt)
                        a2_t = cos(rt - at, st - at)
                        a3_t = cos(rt - st, at - st)
                    l1, l2, l3 = cp.checkpoint(
                        _tri_chunk, rs, ss, rt, at, st,
                        a1_t, a2_t, a3_t, wc, use_reentrant=False)
                    sums[0] = sums[0] + l1
                    sums[1] = sums[1] + l2
                    sums[2] = sums[2] + l3
                    wsum = wsum + wc.sum() * (s1 - s0) * topk
        denom = wsum.clamp_min(1e-8) if torch.is_tensor(wsum) else max(wsum, 1e-8)
        losses = [s / denom for s in sums]
        loss = losses[0] + losses[1] + losses[2]
        return loss, {"r2_tri": float(loss), "r2_a1": float(losses[0]),
                      "r2_a2": float(losses[1]), "r2_a3": float(losses[2])}


@torch.no_grad()
def ema_teacher_cache(student, params, ema_params, images8, patch_hw):
    """Recompute a pair's teacher cache with EMA weights swapped into the
    student's trainable (LoRA) params; heads are the shared frozen ones.
    Live params are restored afterwards. Reuses M.cache_teacher_pair."""
    base = M.get_base_vggt(student)
    backup = [p.detach().clone() for p in params]
    for p, e in zip(params, ema_params):
        p.copy_(e)
    was_training = student.training
    student.eval()
    cache = M.cache_teacher_pair(base, images8, patch_hw)
    if was_training:
        student.train()
    for p, b in zip(params, backup):
        p.copy_(b)
    return cache


def compute_loss(arm, a1, base, teacher_cache, feats24_s, preds, patch_hw, step=0, patch_mask=None, ctm_j=None):
    if arm in ("FM_zero", "FM_mean", "XCO"):
        arm = "C2_b5_rel"  # input-side variants share the C2 objective
    if arm == "A1_deployed":
        return a1(teacher_cache, feats24_s)
    if arm == "A2_projected":
        return loss_a2_projected(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "A3_outdepth":
        return loss_a3_outdepth(teacher_cache, preds)
    if arm == "A4_pose":
        return loss_a4_pose(teacher_cache, preds)
    if arm == "B1_norm_patch":
        return loss_b_patchform(base.depth_head, teacher_cache, feats24_s, patch_hw,
                                space="norm", cam_token=False)
    if arm == "B2_deployed_camtok":
        loss, extra = a1(teacher_cache, feats24_s)
        cs = feats24_s[23][:, :, 0, :].float()
        ct = teacher_cache["feats"][23][:, STUDENT_INDICES, 0, :].float()
        cam = torch.mean((cs - ct) ** 2)
        return loss + cam, {**extra, "camtok": float(cam)}
    if arm == "B3_raw_patch":
        return loss_b_patchform(base.depth_head, teacher_cache, feats24_s, patch_hw,
                                space="raw", cam_token=False)
    if arm == "B4_norm_camtok":
        return loss_b_patchform(base.depth_head, teacher_cache, feats24_s, patch_hw,
                                space="norm", cam_token=True)
    if arm == "B5_conf":
        return loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "B6_norm_cf":
        loss, extra = loss_b_patchform(base.depth_head, teacher_cache, feats24_s, patch_hw,
                                       space="norm", cam_token=False)
        cf, cf_extra = a1.cf_terms(teacher_cache, feats24_s)
        return loss + cf, {**extra, **cf_extra}
    if arm == "C1_pose_rel":
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        return loss_pose_rel(preds["pose_enc"], pt)
    if arm == "C2_b5_rel":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + rel, {**extra, **rel_extra}
    if arm == "C2P_permgate":
        feat, extra = loss_b5_confperm(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + rel, {**extra, **rel_extra}
    if arm == "C2S_studconf":
        w = student_patch_conf(preds, patch_hw)
        feat = _b5_form_with_w(base.depth_head, teacher_cache, feats24_s, patch_hw, w)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + rel, {"conf_w_max": float(w.max()), "conf_w_min": float(w.min()), **rel_extra}
    if arm == "C2SI_invconf":
        conf = student_patch_conf(preds, patch_hw)
        w = (1.0 / (conf + 0.1)).detach()
        w = (w / w.mean().clamp_min(1e-8)).detach()
        feat = _b5_form_with_w(base.depth_head, teacher_cache, feats24_s, patch_hw, w)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + rel, {"conf_w_max": float(w.max()), "conf_w_min": float(w.min()), **rel_extra}
    if arm == "C2O_ohem":
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for layer in TAP_LAYERS:
                hs = M.to_norm(base.depth_head, M.to_patch(feats24_s[layer].float()))
                ht = M.to_norm(base.depth_head, M.to_patch(teacher_cache["feats"][layer][:, STUDENT_INDICES].float()))
                tot = tot + (hs - ht).abs().mean(dim=-1)
                cnt += 1
            w = (tot / cnt)
            w = (w / w.mean().clamp_min(1e-8)).detach()
        feat = _b5_form_with_w(base.depth_head, teacher_cache, feats24_s, patch_hw, w)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + rel, {"conf_w_max": float(w.max()), "conf_w_min": float(w.min()), **rel_extra}
    if arm == "G1_oracle_gate":
        return loss_g1_oracle(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "G2_interp50":
        return loss_g2_interp(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "G4_conf_l23only":
        return loss_g4_l23only(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "D1_relfeat":
        return loss_d1_relfeat(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "D2_layermean":
        return loss_d2_layermean(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "D3_spatialpool":
        return loss_d3_spatialpool(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "D4_depthgrad":
        return loss_d4_depthgrad(teacher_cache, preds)
    if arm == "C4_selfmix":
        return loss_c4_selfmix(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "B5_R1_l0.1":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 0.1 * tri, {**extra, **tri_extra}
    if arm == "B5_R1_l0.3":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 0.3 * tri, {**extra, **tri_extra}
    if arm == "B5_R1_l1.0":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 1.0 * tri, {**extra, **tri_extra}
    if arm == "B5_R1_l10":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 10.0 * tri, {**extra, **tri_extra}
    if arm == "B5_R1_l50":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 50.0 * tri, {**extra, **tri_extra}
    if arm == "B5_R2_l0.3":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r2_trifix(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 0.3 * tri, {**extra, **tri_extra}
    if arm == "B5_R2_l1.0":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r2_trifix(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 1.0 * tri, {**extra, **tri_extra}
    if arm == "B5_R3_l0.3":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        tri, tri_extra = loss_r3_shared(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + 0.3 * tri, {**extra, **tri_extra}
    if arm == "C2_R1_l0.3":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + rel + 0.3 * tri, {**extra, **rel_extra, **tri_extra}
    if arm == "C2_R1_l50":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + rel + 50.0 * tri, {**extra, **rel_extra, **tri_extra}
    if arm == "C2_R2_l0.3":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        tri, tri_extra = loss_r2_trifix(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + rel + 0.3 * tri, {**extra, **rel_extra, **tri_extra}
    if arm == "E2_out":
        e2, e2_extra = loss_e2_outdepth_se(teacher_cache, preds)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return e2 + rel, {**e2_extra, **rel_extra}
    if arm == "C2_E2":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        e2, e2_extra = loss_e2_outdepth_se(teacher_cache, preds)
        return feat + rel + 0.3 * e2, {**extra, **rel_extra, **e2_extra}
    if arm == "CFonly_R2":
        tri, tri_extra = loss_r2_trifix(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return tri, tri_extra
    if arm == "Xpose_only":
        pt8 = teacher_cache["pose_enc8"].float()
        return loss_pose_rel_x(preds["pose_enc"], pt8)
    if arm == "B5_margin":
        return loss_b5_margin(base.depth_head, teacher_cache, feats24_s, patch_hw)
    if arm == "CamRel_only":
        return loss_camrel(base.depth_head, base.camera_head, teacher_cache, feats24_s)
    if arm == "CamTokHC_only":
        return loss_camtok_hc(base.camera_head, teacher_cache, feats24_s)
    if arm == "CamTokHC4_only":
        return loss_camtok_hc(base.camera_head, teacher_cache, feats24_s, layers=[4, 11, 17, 23])
    if arm == "C3":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        ch, ch_extra = loss_camtok_hc(base.camera_head, teacher_cache, feats24_s,
                                      layers=[23])
        a3, a3_extra = loss_a3_outdepth(teacher_cache, preds)
        return feat + rel + 0.3 * ch + 0.3 * a3, {**extra, **rel_extra, **ch_extra, **a3_extra}
    if arm == "C3_ch1":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        ch, ch_extra = loss_camtok_hc(base.camera_head, teacher_cache, feats24_s,
                                      layers=[23])
        a3, a3_extra = loss_a3_outdepth(teacher_cache, preds)
        return feat + rel + 1.0 * ch + 0.3 * a3, {**extra, **rel_extra, **ch_extra, **a3_extra}
    if arm == "C3r":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        cr, cr_extra = loss_camrel(base.depth_head, base.camera_head, teacher_cache, feats24_s)
        a3, a3_extra = loss_a3_outdepth(teacher_cache, preds)
        return feat + rel + 0.3 * cr + 0.3 * a3, {**extra, **rel_extra, **cr_extra, **a3_extra}
    if arm == "C2_CamTokHC":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        ch, ch_extra = loss_camtok_hc(base.camera_head, teacher_cache, feats24_s)
        return feat + rel + 0.3 * ch, {**extra, **rel_extra, **ch_extra}
    if arm == "C2_CamRel":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        cr, cr_extra = loss_camrel(base.depth_head, base.camera_head, teacher_cache, feats24_s)
        return feat + rel + 0.3 * cr, {**extra, **rel_extra, **cr_extra}
    if arm == "Xpose2_only":
        return loss_pose_rel_x2(preds["pose_enc"], teacher_cache["pose_enc8"].float())
    if arm == "C2_xpose2":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        x2, x2_extra = loss_pose_rel_x2(preds["pose_enc"], teacher_cache["pose_enc8"].float(), w_sx=0.5)
        return feat + x2, {**extra, **x2_extra}
    if arm == "B5_maskdistill":
        return loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
    if arm in ("MD25", "MD75", "MDHI", "MDLO", "MDR", "MDASY"):
        return loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
    if arm == "CONFD":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        cs = preds["conf"].float()
        ct = teacher_cache["conf4"].float()
        if cs.dim() == 5:
            cs = cs.squeeze(2)
        if ct.dim() == 5:
            ct = ct.squeeze(2)
        confd = torch.nn.functional.smooth_l1_loss(
            torch.log(cs.clamp_min(1e-3)), torch.log(ct.clamp_min(1e-3)).detach())
        return feat + 0.3 * confd, {**extra, "confd": float(confd)}
    if arm == "SS8M":
        feat, extra = loss_ss8m(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], teacher_cache["pose_enc8"].float())
        return feat + rel, {**extra, **rel_extra}
    if arm == "C2M_maskrel":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + rel, {**extra, **rel_extra}
    if arm == "C2M_ABS":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        abs_l, abs_extra = loss_abs_pose_norm(preds["pose_enc"], teacher_cache)
        return feat + ABS_POSE_W * abs_l, {**extra, **abs_extra}
    if arm == "C2M_ABS_REL":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        abs_l, abs_extra = loss_abs_pose_norm(preds["pose_enc"], teacher_cache)
        return feat + rel + ABS_POSE_W * abs_l, {**extra, **rel_extra, **abs_extra}
    if arm == "C2M_ABSR":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        abs_l, abs_extra = loss_abs_pose_raw(preds["pose_enc"],
                                             teacher_cache["pose_enc8"][:, STUDENT_INDICES].float())
        return feat + ABS_RAW_W * abs_l, {**extra, **abs_extra}
    if arm == "C2M_ABSW1":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        abs_l, abs_extra = loss_abs_pose_norm(preds["pose_enc"], teacher_cache)
        return feat + 1.0 * abs_l, {**extra, **abs_extra}
    if arm == "C2M_ABSR5":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        abs_l, abs_extra = loss_abs_pose_raw(preds["pose_enc"],
                                             teacher_cache["pose_enc8"][:, STUDENT_INDICES].float())
        return feat + 5.0 * abs_l, {**extra, **abs_extra}
    if arm == "C2M_RELAT":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        at, at_extra = loss_abs_t_fl(preds["pose_enc"], pt)
        return feat + rel + 1.0 * at, {**extra, **rel_extra, **at_extra}
    if arm == "C2M_RABS1":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        abs_l, abs_extra = loss_abs_pose_raw(preds["pose_enc"], pt)
        return feat + rel + 1.0 * abs_l, {**extra, **rel_extra, **abs_extra}
    if arm == "C2M_RABS5":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        abs_l, abs_extra = loss_abs_pose_raw(preds["pose_enc"], pt)
        return feat + rel + 5.0 * abs_l, {**extra, **rel_extra, **abs_extra}
    if arm == "C2M_RABS3":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        abs_l, abs_extra = loss_abs_pose_raw(preds["pose_enc"], pt)
        return feat + rel + 3.0 * abs_l, {**extra, **rel_extra, **abs_extra}
    if arm == "C2M_XSH":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        return feat + rel + 0.5 * rkd, {**extra, **rel_extra, **rkd_extra}
    if arm == "C2M_XAC":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        xac, xac_extra = loss_xac_camtok(base.camera_head, teacher_cache, feats24_s)
        return feat + rel + 0.5 * xac, {**extra, **rel_extra, **xac_extra}
    if arm == "C2M_XAC2":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        xac, xac_extra = loss_xac2_camtok(base.camera_head, teacher_cache, feats24_s)
        return feat + rel + 0.5 * xac, {**extra, **rel_extra, **xac_extra}
    if arm == "C2M_XSH1":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        return feat + rel + 1.0 * rkd, {**extra, **rel_extra, **rkd_extra}
    if arm == "C2M_XSHS":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        sc, sc_extra = loss_scale_gauge(preds["pose_enc"], pt)
        return feat + rel + 0.5 * rkd + 0.3 * sc, {**extra, **rel_extra, **rkd_extra, **sc_extra}
    if arm == "C2M_XSH1S":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        sc, sc_extra = loss_scale_gauge(preds["pose_enc"], pt)
        return feat + rel + 1.0 * rkd + 0.3 * sc, {**extra, **rel_extra, **rkd_extra, **sc_extra}
    if arm == "C2M_NREL":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        return feat + 0.5 * rkd, {**extra, **rkd_extra}
    if arm == "C2M_RKD15":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        return feat + 1.5 * rkd, {**extra, **rkd_extra}
    if arm == "C2M_RKDC":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + 1.5 * rkd + 0.3 * cp, {**extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RKDC1":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + 1.5 * rkd + 1.0 * cp, {**extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RKDC1R":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + rel + 1.5 * rkd + 1.0 * cp, {**extra, **rel_extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RKDC1A":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        abs_l, abs_extra = loss_abs_pose_raw(preds["pose_enc"], pt)
        return feat + 1.5 * rkd + 1.0 * cp + 1.0 * abs_l, {**extra, **rkd_extra, **cp_extra, **abs_extra}
    if arm == "C2M_RKDC1H":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose_huber(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + 1.5 * rkd + 1.0 * cp, {**extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RKDCR1H":
        # rkdc1h + corrected rel pose (E_i @ inv(E_j) construction, 2026-09-18)
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose_huber(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + 1.5 * rkd + 1.0 * cp + 1.0 * rel, \
            {**extra, **rkd_extra, **cp_extra, **rel_extra}
    if arm == "C2M_RKDC1HC":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose_huber(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        ctk, ctk_extra = loss_camtok_hc(base.camera_head, teacher_cache, feats24_s,
                                        layers=TAP_LAYERS)
        return feat + 1.5 * rkd + 1.0 * cp + CTK_W * ctk, \
            {**extra, **rkd_extra, **cp_extra, **ctk_extra}
    if arm == "C2M_maskrel_CTK":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        ctk, ctk_extra = loss_camtok_hc(base.camera_head, teacher_cache, feats24_s,
                                        layers=TAP_LAYERS)
        return feat + rel + CTK_W * ctk, {**extra, **rel_extra, **ctk_extra}
    if arm == "C2M_RKLH":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_local_huber(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + rel + 1.5 * rkd + 1.0 * cp, {**extra, **rel_extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RELCH":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_shared_pose_huber(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + rel + 1.5 * rkd + 1.0 * cp, {**extra, **rel_extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RELC":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + rel + 1.0 * cp, {**extra, **rel_extra, **cp_extra}
    if arm in ("C2M_RKDC2", "C2M_RKDC3"):
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        cw = 2.0 if arm == "C2M_RKDC2" else 3.0
        return feat + 1.5 * rkd + cw * cp, {**extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RKCX":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        xa, xa_extra = loss_xac2_camtok(base.camera_head, teacher_cache, feats24_s)
        return feat + 1.5 * rkd + 0.3 * cp + 0.3 * xa, {**extra, **rkd_extra, **cp_extra, **xa_extra}
    if arm == "C2M_RKDCX":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        xa, xa_extra = loss_xac2_camtok(base.camera_head, teacher_cache, feats24_s)
        return feat + 1.5 * rkd + 1.0 * cp + 0.3 * xa, {**extra, **rkd_extra, **cp_extra, **xa_extra}
    if arm == "C2M_RKDCX3":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        xa, xa_extra = loss_xac2_camtok(base.camera_head, teacher_cache, feats24_s)
        return feat + 1.5 * rkd + 1.0 * cp + 3.0 * xa, {**extra, **rkd_extra, **cp_extra, **xa_extra}
    if arm == "C2M_XAP":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        xap, xap_extra = loss_xap_pool(base.depth_head, teacher_cache, feats24_s)
        return feat + rel + 0.5 * xap, {**extra, **rel_extra, **xap_extra}
    if arm == "C2M_RKDT16":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple16(preds["pose_enc"], preds["depth"],
                                     teacher_cache["pose_enc8"].float(),
                                     teacher_cache["depth4"], teacher_cache.get("conf4"))
        return feat + 1.5 * rkd + 1.0 * cp, {**extra, **rkd_extra, **cp_extra}
    if arm == "C2M_RKDCL":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        clo, clo_extra = loss_closure_align(preds["pose_enc"], pt)
        return feat + 1.5 * rkd + 1.0 * cp + 0.5 * clo, {**extra, **rkd_extra, **cp_extra, **clo_extra}
    if arm == "C2M_TGM":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                   teacher_cache["depth4"], teacher_cache.get("conf4"))
        gm, gm_extra = loss_gram_pool(base.depth_head, teacher_cache, feats24_s)
        return feat + 1.5 * rkd + 1.0 * cp + 0.5 * gm, {**extra, **rkd_extra, **cp_extra, **gm_extra}
    if arm == "C2M_RKDS":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        sc, sc_extra = loss_scale_gauge(preds["pose_enc"], pt)
        return feat + 1.5 * rkd + 0.3 * sc, {**extra, **rkd_extra, **sc_extra}
    if arm == "C2M_XEXT":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rkd, rkd_extra = loss_rkd_ext_pose(preds["pose_enc"], teacher_cache["pose_enc8"].float(),
                                           STUDENT_INDICES)
        return feat + 1.5 * rkd, {**extra, **rkd_extra}
    if arm == "C2M_XSHA":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
        abs_l, abs_extra = loss_abs_pose_raw(preds["pose_enc"], pt)
        return feat + rel + 0.5 * rkd + 1.0 * abs_l, {**extra, **rel_extra, **rkd_extra, **abs_extra}
    if arm == "C2M_CamRel":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        cr, cr_extra = loss_camrel(base.depth_head, base.camera_head, teacher_cache, feats24_s)
        return feat + rel + 0.3 * cr, {**extra, **rel_extra, **cr_extra}
    if arm == "CONFD_REL":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        cs = preds["conf"].float()
        ct = teacher_cache["conf4"].float()
        if cs.dim() == 5:
            cs = cs.squeeze(2)
        if ct.dim() == 5:
            ct = ct.squeeze(2)
        confd = torch.nn.functional.smooth_l1_loss(
            torch.log(cs.clamp_min(1e-3)), torch.log(ct.clamp_min(1e-3)).detach())
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + rel + 0.3 * confd, {**extra, **rel_extra, "confd": float(confd)}
    if arm == "C2M_CTM":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        cam, cam_extra = loss_ctm_camtok_j(base.camera_head, teacher_cache, feats24_s, ctm_j)
        return feat + rel + cam, {**extra, **rel_extra, **cam_extra}
    if arm == "B5_CTK":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        ctk, ctk_extra = loss_camtok_hc(base.camera_head, teacher_cache, feats24_s)
        return feat + ctk, {**extra, **ctk_extra}
    if arm == "C2M_CTK":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        ctk, ctk_extra = loss_camtok_hc(base.camera_head, teacher_cache, feats24_s)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel_cap(preds["pose_enc"], pt)
        return feat + ctk + rel, {**extra, **ctk_extra, **rel_extra}
    if arm == "C2M_TRIF3":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        tri, tri_extra = loss_triangle_camtok_v3(base.camera_head, teacher_cache, feats24_s)
        return feat + rel + tri, {**extra, **rel_extra, **tri_extra}
    if arm == "C2M_TRIF2":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        tri, tri_extra = loss_triangle_camtok_v2(base.camera_head, teacher_cache, feats24_s)
        return feat + rel + tri, {**extra, **rel_extra, **tri_extra}
    if arm == "C2M_CONFP":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel_confp(preds["pose_enc"], pt, teacher_cache["conf4"])
        return feat + rel, {**extra, **rel_extra}
    if arm == "C2M_GATE":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        g = float(teacher_cache.get("pose_gate", 1.0))
        return feat + g * rel, {**extra, **rel_extra, "pose_gate": g}
    if arm == "C2M_MC":  # handled in the training loop (needs two forwards)
        raise RuntimeError("C2M_MC is loop-level")
    if arm == "C2M_SCL":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel_scale(preds["pose_enc"], pt)
        return feat + rel, {**extra, **rel_extra}
    if arm == "C2M_CYC":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        cyc, cyc_extra = loss_pose_cycle(preds["pose_enc"])
        return feat + rel + cyc, {**extra, **rel_extra, **cyc_extra}
    if arm == "C2M_HARD":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel_hard(preds["pose_enc"], pt)
        return feat + rel, {**extra, **rel_extra}
    if arm == "C2M_TRIP":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        tri, tri_extra = loss_triangle_pose(preds["pose_enc"], teacher_cache["pose_enc8"].float())
        return feat + rel + tri, {**extra, **rel_extra, **tri_extra}
    if arm == "C2M_TRIF":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        tri, tri_extra = loss_triangle_camtok(base.camera_head, teacher_cache, feats24_s)
        return feat + rel + tri, {**extra, **rel_extra, **tri_extra}
    if arm == "CTM_ANC":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        cam, cam_extra = loss_ctm_camtok_j(base.camera_head, teacher_cache, feats24_s, ctm_j)
        cr, cr_extra = loss_camrel(base.depth_head, base.camera_head, teacher_cache, feats24_s)
        return feat + rel + cam + 0.3 * cr, {**extra, **rel_extra, **cam_extra, **cr_extra}
    if arm == "C2M_REL2":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + 2.0 * rel, {**extra, **rel_extra}
    if arm == "C2M_REL10":
        feat, extra = loss_b5_maskdistill(base.depth_head, teacher_cache, feats24_s, patch_hw, patch_mask)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        return feat + 10.0 * rel, {**extra, **rel_extra}
    if arm == "C2_SP_l01":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        sp, sp_extra = loss_d1_relfeat(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + rel + 0.1 * sp, {**extra, **rel_extra, **sp_extra}
    if arm == "C2_SP_l03":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        sp, sp_extra = loss_d1_relfeat(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return feat + rel + 0.3 * sp, {**extra, **rel_extra, **sp_extra}
    if arm == "C2_A3_l03":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        a3, a3_extra = loss_a3_outdepth(teacher_cache, preds)
        return feat + rel + 0.3 * a3, {**extra, **rel_extra, **a3_extra}
    if arm == "C2_A3_l1":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        a3, a3_extra = loss_a3_outdepth(teacher_cache, preds)
        return feat + rel + 1.0 * a3, {**extra, **rel_extra, **a3_extra}
    if arm == "C2_asym100":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        import math
        lam = 0.5 * (1.0 + math.cos(math.pi * min(step, 30) / 30.0))
        return lam * feat + rel, {"lam_f": lam, **extra, **rel_extra}
    if arm == "C2_xpose":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        xrel, xrel_extra = loss_pose_rel_x(preds["pose_enc"], teacher_cache["pose_enc8"].float())
        return feat + xrel, {**extra, **xrel_extra}
    if arm == "B5_L1723":
        return loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw,
                            layers=[17, 23])
    if arm == "B5_wL17":
        return loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw,
                            layer_w={17: 2.0})
    if arm == "C2_E2_w1":
        feat, extra = loss_b5_conf(base.depth_head, teacher_cache, feats24_s, patch_hw)
        pt = teacher_cache["pose_enc8"][:, STUDENT_INDICES].float()
        rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
        e2, e2_extra = loss_e2_outdepth_se(teacher_cache, preds)
        return feat + rel + 1.0 * e2, {**extra, **rel_extra, **e2_extra}
    if arm == "CFonly_R2_x3":
        tri, tri_extra = loss_r2_trifix(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return 3.0 * tri, tri_extra
    if arm == "CFonly_R1_50":
        tri, tri_extra = loss_r1_qk(base.depth_head, teacher_cache, feats24_s, patch_hw)
        return 50.0 * tri, tri_extra
    raise ValueError(arm)


def save_eval_npz(run_root, exp, scene, depth, ext, intr, gt_ext, gt_intr, image_files, frame_ids, conf=None, dataset=None):
    dataset = dataset or common._CURRENT_DATASET
    export_dir = os.path.join(run_root, "eval32", exp, "model_results",
                              dataset, scene, "unposed")
    os.makedirs(os.path.join(export_dir, "exports", "mini_npz"), exist_ok=True)
    np.savez_compressed(
        os.path.join(export_dir, "exports", "mini_npz", "results.npz"),
        depth=np.round(depth, 8), extrinsics=ext, intrinsics=intr,
        **({"conf": np.round(conf, 3)} if conf is not None else {}))
    np.savez_compressed(
        os.path.join(export_dir, "exports", "gt_meta.npz"),
        extrinsics=gt_ext, intrinsics=gt_intr,
        image_files=np.array(image_files, dtype=object),
        requested_view_count=np.array([32], dtype=np.int32),
        actual_view_count=np.array([len(image_files)], dtype=np.int32),
        sampled_indices=np.array(list(frame_ids), dtype=np.int32))


@torch.no_grad()
def infer_eval32(student, scene_data, eval_frames, device="cuda"):
    """32-view inference with the current in-memory model (stays on GPU)."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    base = M.get_base_vggt(student)
    student.eval()
    imgs = [M.load_image_model(scene_data.image_files[i]) for i in eval_frames]
    arr = np.stack(imgs, 0)
    images = torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", enabled=False):
        preds = base(images)
    depth = preds["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()   # [32,H,W]
    conf = preds.get("depth_conf")
    conf = conf.squeeze(0).float().cpu().numpy() if conf is not None else None
    if conf is not None and conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    pose_enc = preds["pose_enc"]
    if pose_enc.dim() == 2:
        pose_enc = pose_enc.unsqueeze(0)
    H, W = images.shape[-2:]
    ext, intr = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H, W), pose_encoding_type="absT_quaR_FoV")
    return depth, ext.squeeze(0).float().cpu().numpy(), intr.squeeze(0).float().cpu().numpy(), images, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", default="artifacts/diagnostics/bakeoff_v1")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--train_camera_token", action="store_true",
                    help="unfreeze the camera token (C5)")
    ap.add_argument("--full_ft", action="store_true",
                    help="full fine-tune of the aggregator instead of LoRA (C3); lr forced to 1e-5")
    ap.add_argument("--no_ckpt", action="store_true",
                    help="skip saving LoRA checkpoints (disk hygiene)")
    ap.add_argument("--save_conf", action="store_true",
                    help="also save depth_conf in eval npz (needed for conf-filtered TSDF)")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="smoke mode: only N optimizer steps per arm; skips the A0 "
                         "baseline probe/eval32 and all probe/ckpt/eval32 saves (0=off)")
    ap.add_argument("--no_eval32", action="store_true",
                    help="skip eval32 inference/npz at step marks (screening runs)")
    ap.add_argument("--early_stop", action="store_true",
                    help="stop when tail-10 loss mean improves <2%% over prev-10 "
                         "(min max(30, warmup+10) steps); final probe/ckpt/eval32 "
                         "still saved at the stop step")
    ap.add_argument("--two_stage", type=float, default=0.0,
                    help=">0 (e.g. 0.7): sequential curriculum — fixed-slot pairs for "
                         "the first frac*n_steps, then endpoint-anchored pairs "
                         "(kind-tagged manifest required; disables early_stop)")
    ap.add_argument("--sam_rho", type=float, default=0.0,
                    help=">0 enables SAM (two fwd/bwd per step, rho=perturb radius)")
    ap.add_argument("--ema_teacher", type=int, default=0,
                    help="EMA-teacher mode: refresh the current pair's teacher cache "
                         "from EMA student weights every K steps (0=off, LoRA path only)")
    ap.add_argument("--ema_decay", type=float, default=0.995,
                    help="EMA decay for --ema_teacher (ema = decay*ema + (1-decay)*param)")
    ap.add_argument("--teacher_consensus", type=int, default=0,
                    help="consensus teacher: TRAIN pair caches average K 8-view "
                         "contexts (k=0 = the pair's original frames, k>=1 resampled "
                         "extras) via M.cache_teacher_pair_consensus; probe pairs "
                         "keep the single-context cache. 0=off (default)")
    ap.add_argument("--seed", type=int, default=0,
                    help="global RNG seed for LoRA init and per-scene train-order/mask seeds")
    ap.add_argument("--loss_all_pos", action="store_true",
                    help="ablation: keep the 50%% masked student input but compute the "
                         "distill loss on ALL patch positions (default: MGD-style, "
                         "masked positions only)")
    ap.add_argument("--mask_mode", default="image", choices=["image", "token"],
                    help="image = mask input pixels (default); token = clean input, "
                         "zero 50%% of patch tokens at aggregator.patch_embed output "
                         "(feature-level masking, after DINOv2, before aggregator blocks)")
    ap.add_argument("--v2_probe", action="store_true",
                    help="protocol v2: GT-free probe on the manifest probe pairs "
                         "(fixed per-pair masks, robust components vs the frozen "
                         "teacher cache), evaluated at step 0 and every "
                         "--v2_probe_every updates; appended to "
                         "<run_root>/probe_trace/<scene>.jsonl. Coexists with the "
                         "existing GT probe (evaluate_probes), which is untouched.")
    ap.add_argument("--v2_probe_every", type=int, default=10,
                    help="v2 probe cadence in optimizer updates (default 10)")
    ap.add_argument("--v2_ckpt", action="store_true",
                    help="protocol v2: save LoRA at step 0 and every probe point to "
                         "ckpts/<scene>/<arm>/v2/step{N}_lora.pt (hard assert on "
                         "save success). LoRA path only (skipped with --full_ft).")
    ap.add_argument("--v2_rel_weight", type=float, default=0.0,
                    help=">0: append v2_rel_weight * (rot_edges_huber.loss + "
                         "tdir_cos_loss.loss) to the arm loss, teacher detached, "
                         "on the student shared-4 w2c extrinsics (tta_v2 robust "
                         "form; the arm's own rel term, if any, is unchanged)")
    ap.add_argument("--v2_grad_cap", action="store_true",
                    help="requires --v2_rel_weight>0: split backward into two "
                         "fused backwards (base vs v2-rel branch; bit-exact vs "
                         "the normal path), per-param cap on the rel branch — "
                         "first 10 updates measure the median per-param "
                         "||g_R||, afterwards C_R = 4*median is fixed; capped "
                         "rel grads are merged back into p.grad (scaled space, "
                         "GradScaler-compatible) before the existing "
                         "unscale/clip/scaler.step. Records g_base_norm/"
                         "g_R_norm/C_R in the trace extra column.")
    ap.add_argument("--v2_couple_fix", action="store_true",
                    help="use the symmetric valid-mask couple loss (teacher conf "
                         "mask applied to BOTH depth sides, aligned with "
                         "losses.loss_couple_centers) instead of the asymmetric "
                         "abs_pose_loss.loss_couple")
    ap.add_argument("--v2_allpos", action="store_true",
                    help="protocol v2: keep the masked student INPUT but compute "
                         "the maskdistill feature loss on ALL patch positions "
                         "(w = teacher_patch_conf, no *patch_mask) for every arm "
                         "routing through loss_b5_maskdistill; extra records "
                         "allpos=1.0 while mask_ratio still reports the input "
                         "mask. Distinct from --loss_all_pos (loop-level pmask "
                         "replacement, mask_ratio becomes 1.0)")
    ap.add_argument("--swanlab", action="store_true",
                    help="log to swanlab (project free-geometry-tta), one "
                         "experiment per (scene, arm) named {scene}_{arm}{suffix}. "
                         "All logging is exception-guarded: swanlab unavailable "
                         "or offline never affects training.")
    ap.add_argument("--swanlab_suffix", default="",
                    help="experiment-name suffix, e.g. _smoketest (house style: "
                         "protocol tags like _mv13)")
    ap.add_argument("--swanlab_project", default="free-geometry-tta",
                    help="swanlab project name (default free-geometry-tta)")
    ap.add_argument("--swanlab_mode", default=None,
                    help="swanlab init mode override: local | offline | "
                         "disabled (default None = swanlab default)")
    ap.add_argument("--swanlab_logdir", default=None,
                    help="swanlab log_dir override (where local/offline runs "
                         "store their data)")
    ap.add_argument("--v2_baselines_json",
                    default="workspace/protocol_v2/baselines.json",
                    help="scene-level eval comparison source: nested "
                         "{model:{arm:{scene:{auc03,f1,...}}}}; reads vggt / "
                         "baseline / <scene> (and the current arm). Missing "
                         "entries -> <run_root>/scene_eval_pending.json for "
                         "external backfill (run_eval.py runs out-of-process).")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.ema_teacher and args.full_ft:
        raise ValueError("--ema_teacher is only supported on the LoRA path (not --full_ft)")
    if args.ema_teacher and args.teacher_consensus:
        raise ValueError("--ema_teacher and --teacher_consensus are mutually exclusive "
                         "(EMA refresh rebuilds single-context caches)")
    global _V2_COUPLE_FIX, _V2_ALLPOS
    _V2_COUPLE_FIX = bool(args.v2_couple_fix)
    _V2_ALLPOS = bool(args.v2_allpos)
    if args.v2_grad_cap and args.v2_rel_weight <= 0:
        raise ValueError("--v2_grad_cap requires --v2_rel_weight > 0")
    if args.v2_grad_cap and args.sam_rho > 0:
        raise ValueError("--v2_grad_cap and --sam_rho are mutually exclusive "
                         "(both replace the single backward())")
    if args.v2_probe_every < 1:
        raise ValueError("--v2_probe_every must be >= 1")

    manifest = load_manifest(os.path.join(args.run_root, "scene_manifest.json"))
    common.set_dataset(manifest.get("dataset", "scannetpp"))
    # Dynamic student count: mutate the shared list objects in place so every
    # `from common import STUDENT_INDICES` consumer (modeling included) sees it.
    _p0 = next(iter(manifest["scenes"].values()))["train_pairs"][0]
    _n_stu, _n_tch = len(_p0["student_frames"]), len(_p0["teacher_frames"])
    if _n_stu != len(STUDENT_INDICES):
        STUDENT_INDICES[:] = list(range(0, 2 * _n_stu, 2))
        EXTRA_INDICES[:] = [i for i in range(_n_tch) if i not in STUDENT_INDICES]
        print(f"[init] dynamic student slots: {STUDENT_INDICES} of teacher_N={_n_tch}",
              flush=True)
    assert all(p["student_frames"] == [p["teacher_frames"][i] for i in STUDENT_INDICES]
               for sc_ in manifest["scenes"].values()
               for p in sc_["train_pairs"] + sc_["probe_pairs"]), \
        "manifest violates dynamic STUDENT_INDICES layout"
    scenes = args.scenes or sorted(manifest["scenes"])
    device = "cuda"

    trace_path = os.path.join(args.run_root, "training_trace.csv")
    probe_path = os.path.join(args.run_root, "probe_metrics.csv")
    trace_f = open(trace_path, "a", newline="")
    probe_f = open(probe_path, "a", newline="")
    trace_w = csv.DictWriter(trace_f, fieldnames=[
        "scene", "arm", "step", "epoch", "pair_idx", "loss", "lr", "grad_norm", "extra",
        "step_time_s", "peak_mem_mib"])
    probe_w = csv.DictWriter(probe_f, fieldnames=[
        "scene", "arm", "step", "e_depth", "mse_raw", "mse_norm", "mse_proj",
        "probe_auc03", "probe_auc30", "n_valid"])
    if os.path.getsize(trace_path) == 0:
        trace_w.writeheader()
    if os.path.getsize(probe_path) == 0:
        probe_w.writeheader()

    teacher = M.load_teacher(device)

    def fresh_student():
        """New/reset student respecting --train_camera_token and --full_ft."""
        st = M.load_student(device, train_camera_token=args.train_camera_token)
        if args.full_ft:
            base_vggt = st._get_vggt_model()
            for n, p in base_vggt.named_parameters():
                p.requires_grad = ("lora_" not in n) and ("aggregator" in n)
        return st

    student = fresh_student()
    base = M.get_base_vggt(student)
    a1 = DeployedArm()

    for scene in scenes:
        t0 = time.time()
        scene_data = get_scene_data(scene)
        sc = manifest["scenes"][scene]
        eval_frames = sc["eval32_frames"]

        # ---- teacher cache (frozen, once per scene, shared by all arms) ----
        #      + frozen-baseline student pass (for oracle gate / interp target)
        if args.full_ft:
            student = fresh_student()
            base = M.get_base_vggt(student)
        else:
            M.reset_lora_(student)
        student.eval()
        all_pairs = sc["train_pairs"] + sc["probe_pairs"]
        n_train = len(sc["train_pairs"])
        gt_ok = None
        if args.teacher_consensus:
            from common import frames_with_gt_depth
            gt_ok, _ = frames_with_gt_depth(scene)
        pair_images = []
        pair_images8 = []
        caches = []
        for pi_all, pair in enumerate(all_pairs):
            images8, images4 = M.load_pair_images(scene_data, pair["teacher_frames"], device)
            ph, pw = images8.shape[-2] // 14, images8.shape[-1] // 14
            if args.teacher_consensus and pi_all < n_train:
                shared = list(pair["student_frames"])
                pool = [f for f in gt_ok if f not in set(shared)]
                cache = M.cache_teacher_pair_consensus(
                    teacher, scene_data, pair["teacher_frames"], shared, pool,
                    (ph, pw), K=args.teacher_consensus,
                    seed_tag=(scene, pi_all), device=device)
            else:
                cache = M.cache_teacher_pair(teacher, images8, (ph, pw))
            with torch.no_grad():
                f24, psi, preds = M.student_preds(student, images4)
            sdepth = preds["depth"].squeeze(0)
            if sdepth.dim() == 4:
                sdepth = sdepth.squeeze(-1)
            cache["student_feats"] = {l: f24[l].float() for l in TAP_LAYERS}
            cache["interp_target"] = {
                l: 0.5 * (M.to_patch(cache["feats"][l][:, STUDENT_INDICES].float())
                          + M.to_patch(cache["student_feats"][l]))
                for l in TAP_LAYERS}
            gt4 = M.load_probe_gt(scene_data, pair["student_frames"],
                                  (images4.shape[-2], images4.shape[-1]))
            pp_t = perpatch_logres2(
                cache["depth4"].squeeze(0).squeeze(-1).cpu().numpy(), gt4, (ph, pw))
            pp_s = perpatch_logres2(sdepth.cpu().numpy(), gt4, (ph, pw))
            w = torch.from_numpy(np.where(pp_t < pp_s, 1.0, 0.1)).float()
            cache["oracle_w"] = (w / w.mean().clamp_min(1e-8))[None].to(device)  # [1,4,P]
            if pi_all >= n_train:
                cache["pose_disagree_deg"] = rel_rot_deg(
                    preds["pose_enc"], cache["pose_enc8"][:, STUDENT_INDICES].float())
            caches.append(cache)
            pair_images.append(images4)
            pair_images8.append(images8)
        patch_hw = caches[0]["patch_hw"]
        train_caches, probe_caches = caches[:n_train], caches[n_train:]
        train_images = pair_images[:n_train]
        train_images8 = pair_images8[:n_train]
        # ---- protocol v2: GT-free probe contexts from the SAME caches ----
        v2_contexts, v2_overlaps = [], []
        if args.v2_probe or args.v2_ckpt:
            v2_contexts, v2_overlaps = build_v2_probe_contexts(
                scene, sc, probe_caches, pair_images[n_train:], base)
            if any(v2_overlaps):
                print(f"[{scene}] WARNING: probe/train pair key overlap "
                      f"(probe_train_overlap={v2_overlaps})", flush=True)
        if any("pose_disagree_deg" in c for c in probe_caches):
            dg = float(np.median([c["pose_disagree_deg"] for c in probe_caches
                                  if "pose_disagree_deg" in c]))
            pg = 0.0 if dg < 3.0 else 1.0
            for c in train_caches:
                c["pose_gate"] = pg
            print(f"[{scene}] pose_gate={pg:.0f} (median probe-pair disagreement "
                  f"{dg:.2f} deg)", flush=True)
        if args.teacher_consensus:
            print(f"[{scene}] consensus teacher cache K={args.teacher_consensus} "
                  f"({n_train} train pairs, probe pairs single-context) "
                  f"built in {time.time() - t0:.0f}s", flush=True)

        # ---- baseline (A0): fresh student == frozen baseline ----
        if not args.max_steps:
            student.eval()
            probe0 = M.evaluate_probes(student, scene_data, sc["probe_pairs"], probe_caches, device)
            probe_w.writerow(dict(scene=scene, arm="A0_baseline", step=0, **{
                k: probe0[k] for k in probe_w.fieldnames if k in probe0}))
            probe_f.flush()
            if not args.no_eval32:
                depth, ext, intr, _, _conf = infer_eval32(student, scene_data, eval_frames, device)
                save_eval_npz(args.run_root, "A0_baseline", scene, depth, ext, intr,
                              np.asarray(scene_data.extrinsics)[eval_frames],
                              gt_ixt_raw(scene_data, eval_frames),
                              [scene_data.image_files[i] for i in eval_frames], eval_frames,
                              conf=_conf if getattr(args, "save_conf", False) else None)
            print(f"[{scene}] baseline probe e_depth={probe0['e_depth']:.4f} "
                  f"auc03={probe0['probe_auc03']:.4f} (cache+base {time.time()-t0:.0f}s)", flush=True)

        for arm in args.arms:
            if args.full_ft:
                student = fresh_student()
                base = M.get_base_vggt(student)
            else:
                M.reset_lora_(student)
            params = [p for p in student.vggt.parameters() if p.requires_grad] if args.full_ft \
                else student.get_trainable_params()
            lr_arm = 1e-5 if args.full_ft else args.lr
            optimizer = torch.optim.AdamW(params, lr=lr_arm, weight_decay=WD)
            # Protocol v2: keep the 100-step budget fixed regardless of pair count.
            epochs_arm = (max(1, round(100 / n_train)) if "n_train_pairs" in sc
                          else args.epochs)
            n_steps = n_train * epochs_arm
            warm = int(n_steps * WARMUP_RATIO)
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            scheduler = SequentialLR(optimizer, [
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warm),
                CosineAnnealingLR(optimizer, T_max=n_steps - warm, eta_min=1e-8)], [warm])
            scaler = GradScaler(enabled=True)
            if args.v2_grad_cap:
                # GradScaler lazily creates its scale factor on the first
                # scale() call; the split-backward path (v2_grad_cap_backward)
                # never calls scaler.scale(), so init it up front — afterwards
                # scaler.get_scale() is valid in every update.
                scaler.scale(torch.zeros((), device=device))
            torch.cuda.reset_peak_memory_stats()
            step_times = []
            ema_params = None
            if args.ema_teacher:
                ema_params = [p.detach().clone() for p in params]
                print(f"[{scene}] {arm} ema teacher on: K={args.ema_teacher} "
                      f"decay={args.ema_decay} n_params={len(ema_params)}", flush=True)

            # ---- swanlab experiment for this (scene, arm); None when disabled
            #      or init failed — every use below is exception-guarded.
            swan_run = _swan_safe_init(args, scene, arm)

            def _v2_probe_log(out, _run=swan_run):
                _swan_log(_run, _probe_metrics(out), step=int(out["step"]))

            # ---- protocol v2 (flag-gated): GT-free probe evaluator for this
            #      arm (fixed per-pair masks, drawn from stable seeds) + step-0
            #      probe / ckpt of the freshly-reset student.
            v2_evaluator = None
            v2_fwd = None
            v2_grad_state = {"medians": [], "C_R": None}
            if args.v2_probe:
                v2_evaluator = ProbeEvaluator(v2_contexts, model=student)
                v2_fwd = make_v2_probe_forward(student, base)
            if args.v2_probe or args.v2_ckpt:
                student.eval()
                if args.v2_probe:
                    out0 = run_v2_probe(v2_evaluator, v2_fwd, scene, arm, 0,
                                        args.run_root, v2_overlaps, n_train,
                                        log_fn=_v2_probe_log)
                    tot = float(np.mean([r["total"] for r in out0["records"]]))
                    print(f"[{scene}] {arm} v2 probe step=0 records={len(out0['records'])} "
                          f"mean_total={tot:.4f}", flush=True)
                if args.v2_ckpt and not args.full_ft:
                    p0 = save_v2_ckpt(student, args.run_root, scene, arm, 0)
                    print(f"[{scene}] {arm} v2 ckpt step=0 -> {p0}", flush=True)

            for step_mark in EVAL_STEPS:
                pass
            step = 0
            losses = []
            es_stop = False
            epoch = 0
            kinds = [p.get("kind", "fixed") for p in sc["train_pairs"]]
            stage_cut = round(n_steps * args.two_stage) if args.two_stage > 0 else None
            if stage_cut is not None:
                print(f"[{scene}] {arm} two_stage={args.two_stage}: fixed pairs for the "
                      f"first {stage_cut} steps, then endpoint-anchored "
                      f"(fixed={kinds.count('fixed')}, se={kinds.count('se')})", flush=True)
            while True:
                if es_stop:
                    break
                if stage_cut is not None:
                    se_phase = step >= stage_cut
                    pool = [i for i, k in enumerate(kinds) if (k == "se") == se_phase]
                    if not pool:
                        pool = list(range(n_train))
                else:
                    pool = list(range(n_train))
                order = [pool[i] for i in torch.randperm(len(pool), generator=torch.Generator().manual_seed(
                    stable_seed("train_order", scene, epoch, args.seed))).tolist()]
                student.train()
                for pi in order:
                    t_step0 = time.perf_counter()
                    cache = train_caches[pi]
                    images4 = train_images[pi]
                    preds = None  # set by every arm branch below (v2 rel consumes it)
                    if ema_params is not None and step % args.ema_teacher == 0:
                        t_ref = time.perf_counter()
                        images8_r, _ = M.load_pair_images(
                            scene_data, sc["train_pairs"][pi]["teacher_frames"], device)
                        new_cache = ema_teacher_cache(student, params, ema_params,
                                                      images8_r, patch_hw)
                        ddepth = (new_cache["depth4"] - cache["depth4"]).abs().mean()
                        cache.update(new_cache)
                        print(f"[{scene}] {arm} ema refresh step={step + 1} pair={pi} "
                              f"ddepth={float(ddepth):.3e} "
                              f"({time.perf_counter() - t_ref:.2f}s)", flush=True)
                    torch.manual_seed(stable_seed("cf_rng", scene, epoch, pi, args.seed))
                    pmask = None
                    images4_in = images4
                    if arm in ("B5_maskdistill", "C2M_maskrel", "MD25", "MD75", "CONFD",
                               "C2M_CamRel", "CONFD_REL", "B5_CTK", "C2M_CTK", "C2M_REL10", "C2M_REL2", "C2M_TRIP", "C2M_TRIF", "C2M_TRIF2", "C2M_TRIF3", "C2M_HARD", "C2M_SCL", "C2M_CYC", "C2M_CONFP", "C2M_GATE", "C2M_ABS", "C2M_ABS_REL", "C2M_ABSR", "C2M_ABSW1", "C2M_ABSR5", "C2M_RELAT", "C2M_RABS1", "C2M_RABS5", "C2M_RABS3", "C2M_XSH", "C2M_XSHA", "C2M_XAC", "C2M_XAC2", "C2M_XSH1", "C2M_XSHS", "C2M_XSH1S", "C2M_NREL", "C2M_RKD15", "C2M_RKDC", "C2M_RKDS", "C2M_XEXT", "C2M_RKDC1", "C2M_RKDC1R", "C2M_RKDC1A", "C2M_RKDC1H", "C2M_RKDCR1H", "C2M_RKDC1HC", "C2M_maskrel_CTK", "C2M_RKDC2", "C2M_RKDC3", "C2M_RELC", "C2M_RKLH", "C2M_RELCH", "C2M_RKCX", "C2M_RKDCX", "C2M_RKDT16", "C2M_RKDCX3", "C2M_XAP", "C2M_RKDCL", "C2M_TGM"):
                        ratio = {"B5_maskdistill": 0.5, "C2M_maskrel": 0.5, "CONFD": 0.5,
                                 "C2M_CamRel": 0.5, "CONFD_REL": 0.5, "B5_CTK": 0.5,
                                 "C2M_CTK": 0.5, "C2M_REL10": 0.5, "C2M_REL2": 0.5, "C2M_TRIP": 0.5, "C2M_TRIF": 0.5, "C2M_TRIF2": 0.5, "C2M_TRIF3": 0.5, "C2M_HARD": 0.5, "C2M_SCL": 0.5, "C2M_CYC": 0.5, "C2M_CONFP": 0.5, "C2M_GATE": 0.5, "C2M_MC": 0.5, "C2M_ABS": 0.5, "C2M_ABS_REL": 0.5, "C2M_ABSR": 0.5, "C2M_ABSW1": 0.5, "C2M_ABSR5": 0.5, "C2M_RELAT": 0.5, "C2M_RABS1": 0.5, "C2M_RABS5": 0.5, "C2M_RABS3": 0.5, "C2M_XSH": 0.5, "C2M_XSHA": 0.5, "C2M_XAC": 0.5, "C2M_XAC2": 0.5, "C2M_XSH1": 0.5, "C2M_XSHS": 0.5, "C2M_XSH1S": 0.5, "C2M_NREL": 0.5, "C2M_RKD15": 0.5, "C2M_RKDC": 0.5, "C2M_RKDS": 0.5, "C2M_XEXT": 0.5, "C2M_RKDC1": 0.5, "C2M_RKDC1R": 0.5, "C2M_RKDC1A": 0.5, "C2M_RKDC1H": 0.5, "C2M_RKDCR1H": 0.5, "C2M_RKDC1HC": 0.5, "C2M_maskrel_CTK": 0.5, "C2M_RKDC2": 0.5, "C2M_RKDC3": 0.5, "C2M_RELC": 0.5, "C2M_RKLH": 0.5, "C2M_RELCH": 0.5, "C2M_RKCX": 0.5, "C2M_RKDCX": 0.5, "C2M_RKDT16": 0.5, "C2M_RKDCX3": 0.5, "C2M_XAP": 0.5, "C2M_RKDCL": 0.5, "C2M_TGM": 0.5,
                                 "MD25": 0.25, "MD75": 0.75}[arm]
                        images4_in, pmask = mask_image_blocks(
                            images4, ratio, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                    elif arm == "MDR":
                        gr = torch.Generator().manual_seed(stable_seed("maskr", scene, epoch, pi, args.seed))
                        ratio = 0.3 + 0.45 * float(torch.rand((), generator=gr))
                        images4_in, pmask = mask_image_blocks(
                            images4, ratio, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                    elif arm in ("MDHI", "MDLO"):
                        images4_in, pmask = mask_by_conf(
                            images4, cache["conf4"], patch_hw,
                            mode="hi" if arm == "MDHI" else "lo")
                    elif arm == "MDASY":
                        images4_in, pmask = mask_per_view_asym(
                            images4, patch_hw,
                            torch.Generator().manual_seed(
                                stable_seed("maskasy", scene, epoch, pi, args.seed)))
                    if args.loss_all_pos and pmask is not None:
                        # ablation: keep the masked INPUT but supervise ALL patch
                        # positions (standard KD-style; contrasts with the default
                        # MGD-style masked-positions-only supervision)
                        pmask = torch.ones_like(pmask)
                    _feat_hook = None
                    if args.mask_mode == "token" and pmask is not None:
                        # Token-level masking: clean input (undo image masking),
                        # install feature hook to zero patch tokens at
                        # aggregator.patch_embed output (after DINOv2, before
                        # aggregator blocks). Uses the 50% corruption mask.
                        images4_in = images4  # clean input
                        _corruption = None
                        # Re-draw the corruption mask (same seed as image mode)
                        gen2 = torch.Generator(device=images4.device).manual_seed(
                            stable_seed("mask", scene, epoch, pi, args.seed))
                        _ph, _pw = patch_hw
                        _corruption = (torch.rand(images4.shape[1], _ph * _pw,
                                        generator=gen2, device=images4.device) < 0.5).float()
                        _corruption = _corruption.reshape(1, images4.shape[1], _ph * _pw)
                        agg = student._get_aggregator()
                        _feat_hook = agg.patch_embed.register_forward_hook(
                            _feat_mask_hook(_corruption))
                    if arm in ("FM_zero", "FM_mean"):
                        images8m = train_images8[pi].clone()
                        if arm == "FM_zero":
                            images8m[:, EXTRA_INDICES] = 0.0
                        else:
                            images8m[:, EXTRA_INDICES] = (
                                train_images8[pi][:, STUDENT_INDICES].mean(dim=1, keepdim=True))
                        with autocast(enabled=True):
                            f8, psi, p8 = M.student_preds(student, images8m)
                            feats24_s = [f[:, STUDENT_INDICES].contiguous() for f in f8]
                            preds = {k: (v[:, STUDENT_INDICES].contiguous()
                                         if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == 8
                                         else v) for k, v in p8.items()}
                            loss, extra = compute_loss("C2_b5_rel", a1, base, cache,
                                                       feats24_s, preds, patch_hw, step=step)
                    elif arm == "XCO":
                        imagesB = train_images8[pi][:, XCO_B].contiguous()
                        with autocast(enabled=True):
                            featsA, psi, predsA = M.student_preds(student, images4)
                            featsB, _, _ = M.student_preds(student, imagesB)
                            loss, extra = compute_loss("C2_b5_rel", a1, base, cache,
                                                       featsA, predsA, patch_hw, step=step)
                            xco, xco_extra = loss_xco_consistency(base.depth_head, featsA, featsB)
                            loss = loss + xco
                            extra = {**extra, **xco_extra}
                        preds = predsA  # v2 rel branch consumes preds["pose_enc"]
                    elif arm == "SS8M":
                        images8m, pmask8 = mask_image_blocks(
                            train_images8[pi], 0.5, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask8", scene, epoch, pi, args.seed)))
                        with autocast(enabled=True):
                            feats8_s, psi, preds = M.student_preds(student, images8m)
                            loss, extra = compute_loss("SS8M", a1, base, cache,
                                                       feats8_s, preds, patch_hw, step=step,
                                                       patch_mask=pmask8)
                    elif arm == "MDK2":
                        with autocast(enabled=True):
                            loss, extra = 0.0, {}
                            for k in range(2):
                                im_k, pm_k = mask_image_blocks(
                                    images4, 0.5, patch_hw,
                                    torch.Generator(device=images4.device).manual_seed(
                                        stable_seed("mask", scene, epoch, pi, k, args.seed)))
                                f_k, _, p_k = M.student_preds(student, im_k)
                                l_k, e_k = compute_loss(
                                    "B5_maskdistill", a1, base, cache, f_k, p_k,
                                    patch_hw, step=step, patch_mask=pm_k)
                                loss = loss + l_k / 2.0
                                extra = e_k
                            preds = p_k  # v2 rel branch consumes preds["pose_enc"]
                    elif arm == "MDF":
                        _, pmask = mask_image_blocks(
                            images4, 0.5, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                        agg = student._get_aggregator()
                        hook = agg.patch_embed.register_forward_hook(_feat_mask_hook(pmask))
                        try:
                            with autocast(enabled=True):
                                feats24_s, psi, preds = M.student_preds(student, images4)
                                loss, extra = compute_loss(
                                    "B5_maskdistill", a1, base, cache, feats24_s, preds,
                                    patch_hw, step=step, patch_mask=pmask)
                        finally:
                            hook.remove()
                    elif arm == "CTM":
                        gc = torch.Generator().manual_seed(
                            stable_seed("ctm", scene, epoch, pi, args.seed))
                        j = int(torch.randint(4, (1,), generator=gc))
                        agg = student._get_aggregator()

                        def _cam_prehook(module, args_):
                            tokens = args_[0].clone()
                            tokens[j, 0, :] = 0.0
                            return (tokens,) + args_[1:]

                        hook = agg.frame_blocks[0].register_forward_pre_hook(_cam_prehook)
                        try:
                            with autocast(enabled=True):
                                feats24_s, psi, preds = M.student_preds(student, images4)
                                feat, extra = loss_b5_conf(base.depth_head, cache,
                                                           feats24_s, patch_hw)
                                cam, cam_extra = loss_ctm_camtok_j(base.camera_head, cache,
                                                                   feats24_s, j)
                                loss = feat + cam
                                extra = {**extra, **cam_extra}
                        finally:
                            hook.remove()
                    elif arm == "C2M_MC":
                        images4_in, pmask = mask_image_blocks(
                            images4, 0.5, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                        with autocast(enabled=True):
                            feats24_s, psi, preds = M.student_preds(student, images4_in)
                            feat, extra = loss_b5_maskdistill(base.depth_head, cache,
                                                              feats24_s, patch_hw, pmask)
                            pt = cache["pose_enc8"][:, STUDENT_INDICES].float()
                            rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
                            f16, _, _ = M.student_preds(student, train_images8[pi])
                            mc, mc_extra = loss_mc_consistency(base.depth_head, feats24_s, f16)
                            loss = feat + rel + 0.5 * mc
                            extra = {**extra, **rel_extra, **mc_extra}
                    elif arm == "C2M_XRKD":
                        images4_in, pmask = mask_image_blocks(
                            images4, 0.5, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                        extras_pi = [i for i in range(train_images8[pi].shape[1])
                                     if i not in STUDENT_INDICES]
                        aidx = extras_pi[stable_seed("anchor", scene, args.seed) % len(extras_pi)]
                        with autocast(enabled=True):
                            images5 = torch.cat([images4_in, train_images8[pi][:, aidx:aidx + 1]], dim=1)
                            feats24_s, psi, preds = M.student_preds(student, images5)
                            feats4_s = [f[:, :4] for f in feats24_s]
                            feat, extra = loss_b5_maskdistill(base.depth_head, cache,
                                                              feats4_s, patch_hw, pmask)
                            pt = cache["pose_enc8"][:, STUDENT_INDICES].float()
                            rel, rel_extra = loss_pose_rel(preds["pose_enc"][:, :4], pt)
                            rkd, rkd_extra = loss_rkd_triplet_pose(
                                preds["pose_enc"], cache["pose_enc8"].float(),
                                STUDENT_INDICES, 4, aidx)
                            loss = feat + rel + 0.5 * rkd
                            extra = {**extra, **rel_extra, **rkd_extra}
                    elif arm == "C2M_XMIX":
                        images4_in, pmask = mask_image_blocks(
                            images4, 0.5, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                        with autocast(enabled=True):
                            feats24_s, psi, preds = M.student_preds(student, images4_in)
                            feat, extra = loss_b5_maskdistill(base.depth_head, cache,
                                                              feats24_s, patch_hw, pmask)
                            pt = cache["pose_enc8"][:, STUDENT_INDICES].float()
                            rel, rel_extra = loss_pose_rel(preds["pose_enc"], pt)
                            extras_t = [i for i in range(cache["pose_enc8"].shape[1])
                                        if i not in STUDENT_INDICES]
                            aidx = extras_t[stable_seed("anchor", scene, args.seed) % len(extras_t)]
                            rkd, rkd_extra = loss_rkd_mix_pose(
                                preds["pose_enc"], cache["pose_enc8"].float(),
                                STUDENT_INDICES, aidx)
                            loss = feat + rel + 0.5 * rkd
                            extra = {**extra, **rel_extra, **rkd_extra}
                    elif arm == "C2M_PMC":
                        images4_in, pmask = mask_image_blocks(
                            images4, 0.5, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                        with autocast(enabled=True):
                            feats24_s, psi, preds = M.student_preds(student, images4_in)
                            pt = cache["pose_enc8"][:, STUDENT_INDICES].float()
                            feat, extra = loss_b5_maskdistill(base.depth_head, cache,
                                                              feats24_s, patch_hw, pmask)
                            rkd, rkd_extra = loss_rkd_shared_pose(preds["pose_enc"], pt)
                            cp, cp_extra = loss_couple(preds["pose_enc"], preds["depth"], pt,
                                                       cache["depth4"], cache.get("conf4"))
                            _, _, preds16 = M.student_preds(student, train_images8[pi])
                            pmc, pmc_extra = loss_abs_pose_raw(preds16["pose_enc"],
                                                               cache["pose_enc8"].float())
                            loss = feat + 1.5 * rkd + 1.0 * cp + 0.5 * pmc
                            extra = {**extra, **rkd_extra, **cp_extra, **pmc_extra}
                    elif arm in ("C2M_CTM", "CTM_ANC"):
                        images4_in, pmask = mask_image_blocks(
                            images4, 0.5, patch_hw,
                            torch.Generator(device=images4.device).manual_seed(
                                stable_seed("mask", scene, epoch, pi, args.seed)))
                        gc = torch.Generator().manual_seed(
                            stable_seed("ctm", scene, epoch, pi, args.seed))
                        j = int(torch.randint(4, (1,), generator=gc))
                        agg = student._get_aggregator()

                        def _cam_prehook_m(module, args_):
                            tokens = args_[0].clone()
                            tokens[j, 0, :] = 0.0
                            return (tokens,) + args_[1:]

                        hook = agg.frame_blocks[0].register_forward_pre_hook(_cam_prehook_m)
                        try:
                            with autocast(enabled=True):
                                feats24_s, psi, preds = M.student_preds(student, images4_in)
                                loss, extra = compute_loss(
                                    arm, a1, base, cache, feats24_s, preds,
                                    patch_hw, step=step, patch_mask=pmask, ctm_j=j)
                        finally:
                            hook.remove()
                    else:
                        with autocast(enabled=True):
                            feats24_s, psi, preds = M.student_preds(student, images4_in)
                            loss, extra = compute_loss(arm, a1, base, cache, feats24_s, preds, patch_hw, step=step, patch_mask=pmask)
                    if _feat_hook is not None:
                        _feat_hook.remove()
                        _feat_hook = None
                    # ---- protocol v2 (flag-gated): robust rel branch + grad cap.
                    #      With both flags off this block reduces to the original
                    #      single scaler.scale(loss).backward() (regression-locked).
                    loss_base = loss
                    loss_rel = None
                    if args.v2_rel_weight > 0:
                        assert preds is not None and "pose_enc" in preds, \
                            f"--v2_rel_weight requires an arm branch that produces preds (arm={arm})"
                        loss_rel, rel_extra = v2_rel_pose_loss(
                            preds["pose_enc"], cache, images4.shape[-2:],
                            args.v2_rel_weight)
                        loss = loss_base + loss_rel
                        extra = {**extra, **rel_extra}
                    if args.v2_grad_cap:
                        # split backward: two fused backwards (bit-exact vs the
                        # single fused path; autograd.grad is NOT — it computes
                        # weight grads behind the autocast boundary differently),
                        # rel branch per-param-norm-capped, merged into p.grad
                        # in scaled space so unscale_/clip/scaler.step are exact.
                        cap_extra = v2_grad_cap_backward(
                            loss_base, loss_rel, params, scaler, v2_grad_state)
                        extra = {**extra, **cap_extra}
                    else:
                        scaler.scale(loss).backward()
                    if args.sam_rho > 0:
                        # SAM: eps = rho*g/||g|| is invariant to the GradScaler factor,
                        # so it can be computed from scaled grads directly.
                        with torch.no_grad():
                            ps = [p for p in params if p.grad is not None]
                            gnorm = torch.norm(torch.stack([p.grad.norm() for p in ps])) + 1e-12
                            eps = [args.sam_rho * p.grad / gnorm for p in ps]
                            for p, e in zip(ps, eps):
                                p.add_(e)
                        optimizer.zero_grad(set_to_none=True)
                        torch.manual_seed(stable_seed("cf_rng", scene, epoch, pi, args.seed))
                        with autocast(enabled=True):
                            feats24_s, psi, preds = M.student_preds(student, images4_in)
                            loss, extra = compute_loss(arm, a1, base, cache, feats24_s, preds, patch_hw, step=step, patch_mask=pmask)
                        scaler.scale(loss).backward()
                        with torch.no_grad():
                            for p, e in zip(ps, eps):
                                p.sub_(e)
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if ema_params is not None:
                        with torch.no_grad():
                            for e, p in zip(ema_params, params):
                                e.mul_(args.ema_decay).add_(p.detach(), alpha=1.0 - args.ema_decay)
                    step += 1
                    step_dt = time.perf_counter() - t_step0
                    step_times.append(step_dt)
                    peak_mib = torch.cuda.max_memory_allocated() / 2**20
                    trace_w.writerow(dict(
                        scene=scene, arm=arm, step=step, epoch=epoch, pair_idx=pi,
                        loss=float(loss), lr=scheduler.get_last_lr()[0],
                        grad_norm=float(gn), extra=str(extra),
                        step_time_s=f"{step_dt:.4f}", peak_mem_mib=f"{peak_mib:.1f}"))
                    # ---- swanlab: aggregate curve + per-pair curves (x = step)
                    _swan_log(swan_run, {
                        "loss": float(loss),
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(gn),
                        ** _pair_metrics(pi, loss, extra),
                    }, step=step)
                    # ---- protocol v2 (flag-gated): probe + ckpt at probe points.
                    #      Runs BEFORE the max_steps break so smoke mode
                    #      (max_steps=3, probe_every=1) still exercises it.
                    if (args.v2_probe or args.v2_ckpt) \
                            and step % args.v2_probe_every == 0:
                        if args.v2_probe:
                            outp = run_v2_probe(v2_evaluator, v2_fwd, scene, arm,
                                                step, args.run_root, v2_overlaps,
                                                n_train, log_fn=_v2_probe_log)
                            tot = float(np.mean([r["total"] for r in outp["records"]]))
                            print(f"[{scene}] {arm} v2 probe step={step} "
                                  f"records={len(outp['records'])} mean_total={tot:.4f}",
                                  flush=True)
                        if args.v2_ckpt and not args.full_ft:
                            pc = save_v2_ckpt(student, args.run_root, scene, arm, step)
                            print(f"[{scene}] {arm} v2 ckpt step={step} -> {pc}",
                                  flush=True)
                    if args.max_steps and step >= args.max_steps:
                        break
                    losses.append(float(loss))
                    es_fire = False
                    if args.early_stop and not es_stop and stage_cut is None:
                        es_min = max(30, warm + 10)
                        if step >= es_min and len(losses) >= 20:
                            recent = float(np.mean(losses[-10:]))
                            previous = float(np.mean(losses[-20:-10]))
                            if previous - recent < 0.02 * abs(previous):
                                es_fire = True
                    if step in (30, 100) or step == n_steps or es_fire:
                        student.eval()
                        probe = M.evaluate_probes(student, scene_data, sc["probe_pairs"],
                                                  probe_caches, device)
                        probe_w.writerow(dict(scene=scene, arm=arm, step=step, **{
                            k: probe[k] for k in probe_w.fieldnames if k in probe}))
                        probe_f.flush()
                        if not args.no_ckpt and not args.full_ft:
                            ckpt_dir = os.path.join(args.run_root, "ckpts", scene, arm)
                            os.makedirs(ckpt_dir, exist_ok=True)
                            student.save_lora_weights(os.path.join(ckpt_dir, f"step{step}_lora.pt"))
                        if not args.no_eval32:
                            depth, ext, intr, _, _conf = infer_eval32(student, scene_data, eval_frames, device)
                            save_eval_npz(args.run_root, f"{arm}_step{step}", scene, depth, ext, intr,
                                          np.asarray(scene_data.extrinsics)[eval_frames],
                                          gt_ixt_raw(scene_data, eval_frames),
                                          [scene_data.image_files[i] for i in eval_frames], eval_frames,
                                          conf=_conf if getattr(args, "save_conf", False) else None)
                        student.train()
                    if es_fire:
                        print(f"[{scene}] {arm} early_stop at step {step} "
                              f"(tail10 {recent:.4f} vs prev10 {previous:.4f})", flush=True)
                        es_stop = True
                        break
                epoch += 1
                if args.max_steps and step >= args.max_steps:
                    break
                if stage_cut is None and epoch >= epochs_arm:
                    break
                if stage_cut is not None and step >= n_steps:
                    break
            trace_f.flush()
            peak_mib = torch.cuda.max_memory_allocated() / 2**20
            mean_dt = float(np.mean(step_times)) if step_times else float("nan")
            print(f"[{scene}] {arm} summary: steps={step} "
                  f"mean_step_time_s={mean_dt:.3f} peak_mem_mib={peak_mib:.1f}", flush=True)
            # ---- swanlab: selector replay (checkpoint-selection audit) + scene
            #      eval comparison (or pending-eval record for external backfill).
            #      Gate on swan_run: swanlab off/unavailable -> zero side effects.
            _swan_selector_summary(args.run_root, scene, arm, swan_run)
            if swan_run is not None:
                _swan_scene_eval(args, scene, arm, swan_run)
            _swan_finish(swan_run)
            print(f"[{scene}] {arm} done ({time.time()-t0:.0f}s cumulative)", flush=True)

        del caches, pair_images
        torch.cuda.empty_cache()

    trace_f.close()
    probe_f.close()
    print("DONE")


if __name__ == "__main__":
    main()
