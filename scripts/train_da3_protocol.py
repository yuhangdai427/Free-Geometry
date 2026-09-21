#!/usr/bin/env python3
"""DA3-giant-1.1 per-scene TTA — Free-Geometry protocol v1 (C2M: maskdistill + rel-pose).

Port of the verified VGGT protocol (docs/TTA_PROTOCOL_v1_2026-09-13.md) to DA3.
Zero-GT frame selection (N/tau rules), frozen teacher cache, LoRA r32/a32 on all
40 aggregator blocks, AdamW 3e-5 cosine, fixed 100 steps, then pose/depth eval on
the protocol eval frames (benchmark-100 for N>=100, else allv).

Smoke example (2 ScanNet++ scenes):
    python scripts/train_da3_protocol.py \
        --scenes 09c1414f1b 1ada7a0617 \
        --output_root workspace/da3_protocol_smoke \
        --steps 100
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diagnostics", "free_geometry"))

import common as fg_common
from depth_anything_3.test_time_adaption import protocol_v1 as P


def get_scene_data(scene: str):
    """Dispatch via diagnostics.free_geometry.common (set_dataset first): identical
    image_lists as the frozen manifests for all 4 datasets."""
    return fg_common.get_scene_data(scene)


def load_gt_depth_raw(path: str, out_hw) -> np.ndarray:
    """GT depth via the shared 4-dataset convention (common.load_gt_depth)."""
    return fg_common.load_gt_depth(path, out_hw)


def cv2_imread_unchanged(path):
    import cv2

    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def cv2_resize(d, wh):
    import cv2

    return cv2.resize(d, wh, interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def evaluate_scene(student, scene_data, eval_frames, max_frames: int = 0,
                   scene: str = "", dataset_obj=None, export_dir: str = None,
                   use_ray_pose: bool = False) -> dict:
    """ONE inference on the protocol eval frames with the in-memory student (LoRA
    active), producing ALL metrics together: pose AUC (compute_pose, w2c),
    scale-fitted AbsRel / δ1.25, AND recon_unposed F1/CD (mini_npz export ->
    gt_meta -> fuse3d TSDF -> eval3d) when dataset_obj+export_dir are given.
    use_ray_pose=True (2026-09-21): camera from the RAY map (origin-weighted T
    + optimal-rotation R + ray-solved intrinsics) instead of CameraDec —
    extrinsics AND intrinsics in the npz export are ray-solved too."""
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous

    frames = list(eval_frames)
    if max_frames > 0 and len(frames) > max_frames:
        frames = frames[:max_frames]
    image_files = [scene_data.image_files[i] for i in frames]

    student.eval()
    # Accept both StudentModel (has .da3) and raw DepthAnything3 (a0 baseline)
    _model = student.da3 if hasattr(student, 'da3') else student
    pred = _model.inference(
        image=image_files,
        process_res=P.PROCESS_RES,
        process_res_method="upper_bound_resize",
        ref_view_strategy="first",
        export_dir=export_dir,
        export_format="mini_npz" if export_dir else "mini_npz",
        use_ray_pose=use_ray_pose,
    )
    depth = np.asarray(pred.depth, dtype=np.float32)  # [N,H,W]
    ext = np.asarray(pred.extrinsics, dtype=np.float32)  # [N,4,4] w2c

    gt_ext = np.asarray(scene_data.extrinsics)[frames]  # w2c
    pose = compute_pose(
        as_homogeneous(torch.from_numpy(ext).float()),
        as_homogeneous(torch.from_numpy(gt_ext).float()),
    )

    # Depth metrics with one least-squares scale per scene (GT-valid px only).
    # DTU/dtu64 have no gt_depth_files -> skip depth metrics (pose + recon only).
    # NOTE: aux is an addict Dict — missing keys return empty Dict (not None),
    # so hasattr() is always True; must check the VALUE is a real list of strs.
    gt = None
    try:
        _gtf = scene_data.aux['gt_depth_files']
        if _gtf and isinstance(_gtf, (list, tuple)) and isinstance(_gtf[0], str):
            gt = np.stack([
                load_gt_depth_raw(_gtf[i], depth.shape[-2:]) for i in frames
            ])
    except Exception:
        gt = None
    if gt is not None:
        omega = np.isfinite(gt) & (gt > 0) & np.isfinite(depth) & (depth > 0)
    else:
        omega = np.zeros_like(depth, dtype=bool)
    n_valid = int(omega.sum())
    out = {
        "n_eval_frames": len(frames),
        "auc03": float(pose.auc03),
        "auc05": float(pose.auc05),
        "auc15": float(pose.auc15),
        "auc30": float(pose.auc30),
        "n_valid_px": n_valid,
    }
    if n_valid > 0:
        p, g = depth[omega].astype(np.float64), gt[omega].astype(np.float64)
        scale = float((p * g).sum() / max((p * p).sum(), 1e-12))
        ps = p * scale
        abs_rel = float(np.mean(np.abs(ps - g) / g))
        delta125 = float(np.mean(np.maximum(ps / g, g / ps) < 1.25))
        abs_rel_raw = float(np.mean(np.abs(p - g) / g))
        out.update({
            "abs_rel": abs_rel,
            "delta125": delta125,
            "ls_scale": scale,
            "abs_rel_unscaled": abs_rel_raw,
        })

    # ---- recon_unposed F1/CD from the SAME inference's npz export ----
    if dataset_obj is not None and export_dir is not None:
        import zipfile

        result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
        deadline = time.time() + 60
        while True:
            try:
                with zipfile.ZipFile(result_path) as z:
                    if z.testzip() is None:
                        break
            except (FileNotFoundError, EOFError, OSError, zipfile.BadZipFile):
                pass
            if time.time() > deadline:
                raise TimeoutError(f"export timeout: {result_path}")
            time.sleep(0.1)
        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        payload = {
            "extrinsics": np.asarray(scene_data.extrinsics)[frames],
            "intrinsics": np.asarray(scene_data.intrinsics)[frames],
            "image_files": np.array(image_files, dtype=object),
        }
        aux = scene_data.aux
        if getattr(aux, "get", None) and aux.get("mask_files") is not None:
            payload["mask_files"] = np.array([aux["mask_files"][i] for i in frames], dtype=object)
        np.savez_compressed(meta_path, **payload)
        fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        try:
            dataset_obj.fuse3d(scene, result_path, fuse_path, "recon_unposed")
            recon = dataset_obj.eval3d(scene, fuse_path)
        except NotImplementedError:
            recon = {}  # dtu64: pose-only, no reconstruction eval
        for k, v in dict(recon).items():
            out[f"recon_{k}"] = float(v)
    return out


def make_dataset(ds):
    import importlib

    table = {
        "scannetpp": ("depth_anything_3.bench.datasets.scannetpp", "ScanNetPP"),
        "7scenes": ("depth_anything_3.bench.datasets.sevenscenes", "SevenScenes"),
        "hiroom": ("depth_anything_3.bench.datasets.hiroom", "HiRoomDataset"),
        "eth3d": ("depth_anything_3.bench.datasets.eth3d", "ETH3D"),
        "dtu": ("depth_anything_3.bench.datasets.dtu", "DTU"),
        "dtu64": ("depth_anything_3.bench.datasets.dtu64", "DTU64"),
    }
    mod = importlib.import_module(table[ds][0])
    return getattr(mod, table[ds][1])()


def load_scene_baseline(path: str, dataset: str, scene: str):
    """baselines.json layout: {model: {dataset: {arm: {scene: {metric: value},
    "_source": [...], "_n_scenes": ...}}}} — scene keys are interleaved with
    "_" -prefixed meta keys; we read da3.<dataset>.baseline.<scene> and filter.
    Returns the metric dict, or None (with a printed warning) when missing."""
    if not os.path.exists(path):
        print(f"[warn] baselines.json not found: {path}; skipping baseline comparison")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        node = data.get("da3", {}).get(dataset, {}).get("baseline", {})
        entry = node.get(scene)
        if not isinstance(entry, dict):
            print(f"[warn] no baseline for {dataset}/{scene} in {path}; skipping")
            return None
        return {k: v for k, v in entry.items() if not k.startswith("_")}
    except Exception as e:
        print(f"[warn] failed to read baseline for {dataset}/{scene} ({e}); skipping")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3 per-scene TTA (protocol v1, C2M)")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default="scannetpp")
    ap.add_argument("--model_name", default="model_weights/DA3-GIANT-1.1")
    ap.add_argument("--output_root", default="workspace/da3_protocol_smoke")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--freeze_camera_token", action="store_true",
                    help="freeze the camera token (LoRA-only adaptation; raymap-path "
                         "A/B: stop ray/focal drift via the dedicated camera channel)")
    ap.add_argument("--epochs", type=int, default=P.EPOCHS)
    ap.add_argument("--lr", type=float, default=P.LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose_weight", type=float, default=1.0,
                    help="1.0 = C2M (default); 0.0 = pure maskdistill (protocol fine "
                         "variant for tau<=0.55 datasets)")
    ap.add_argument("--arm", default="c2m", choices=["c2m", "pw0", "rkdc1h", "rkdc1hc", "rkdc1hr", "adaptive", "adaptive_a", "adaptive_b", "adaptive_s", "aligned", "aligned_rkd"],
                    help="c2m = maskdistill+rel; pw0 = pure maskdistill; "
                         "rkdc1h = maskdistill + 1.5*rkd_huber + 1.0*couple; "
                         "rkdc1hc = rkdc1h + ctk; rkdc1hr = rkdc1h + corrected rel; "
                         "adaptive = gap-weighted rel + 1.5*rkd + couple; "
                         "adaptive_a = gap-weighted rel + (1-w_rel)*rkd + couple; "
                         "adaptive_b = gap-weighted rel + w_rkd*rkd + couple; "
                         "adaptive_s = gap-gated sim3(Umeyama-aligned pose) + couple; "
                         "aligned = md + sim3(Umeyama-aligned pose) + couple; "
                         "aligned_rkd = aligned + 1.5*rkd shape anchor")
    ap.add_argument("--ratio_mix", default=None,
                    help="comma-separated teacher:student ratios mixed within ONE run, "
                         "e.g. '8:4,16:4,24:8' (SelfEvo-style mixed asymmetry per pair)")
    ap.add_argument("--selfevo", action="store_true",
                    help="SelfEvo-faithful student sampling: teacher-window FIRST+LAST "
                         "anchored, per-pair random L middle frames")
    ap.add_argument("--combo", action="store_true",
                    help="combined sampling: fixed-slot pairs (camera) + "
                         "endpoint-anchored L>=4 pairs (fusion), see --combo_se_frac")
    ap.add_argument("--combo_se_frac", type=float, default=0.5,
                    help="fraction of endpoint-anchored pairs in combo mode "
                         "(0.5 = 1:1, 0.25 = 3:1 fixed:SE)")
    ap.add_argument("--sparse_overlap", action="store_true",
                    help="covisibility-constrained teacher windows on the tau<=0.55 "
                         "random branch: resample until every teacher frame's mean "
                         "SIFT match fraction vs the shared frames >= 0.1 (best of "
                         "40 tries). Fixes wide-baseline windows that mix "
                         "mutually-invisible frames (e.g. eth3d facade)")
    ap.add_argument("--force_dense", action="store_true",
                    help="A/B probe: force the dense_equidistant_sift branch "
                         "regardless of tau (single-scene sampling ablation)")
    ap.add_argument("--rel_weight", type=float, default=1.0,
                    help="weight of the corrected rel-pose term (rkdc1hr arm)")
    ap.add_argument("--ctk_weight", type=float, default=1.0,
                    help="weight of the camera-token KD term (rkdc1hc arm)")
    ap.add_argument("--two_stage", type=float, default=0.0,
                    help=">0 (e.g. 0.7): sequential curriculum — fixed-slot pairs for "
                         "the first frac*steps, then endpoint-anchored pairs (combo "
                         "mode only; disables early_stop)")
    ap.add_argument("--mask_ratio", type=float, default=0.5,
                    help="input patch mask ratio for the distill loss; 0.0 = nomask "
                         "(clean-input distillation at all positions)")
    ap.add_argument("--mask_mode", default="image",
                    choices=["image", "none", "token_shallow", "token_feat"],
                    help="mask-position ablation: image = fill ImageNet-mean on "
                         "input pixels (current default); none = clean input; "
                         "token_shallow = zero patch tokens right after patch "
                         "projection; token_feat = zero patch tokens at "
                         "--mask_layer output (default 12 = before DA3's "
                         "cross-view attention starts at block 13)")
    ap.add_argument("--mask_layer", type=int, default=12,
                    help="block index for mask_mode=token_feat (output hook)")
    ap.add_argument("--loss_all_pos", action="store_true",
                    help="ablation: keep the masked student input but compute the "
                         "distill loss on ALL patch positions (default: MGD-style, "
                         "masked positions only)")
    ap.add_argument("--abs_pose_w", type=float, default=0.0,
                    help="weight of absolute pose distillation (R chordal + t Huber) "
                         "added to adaptive arms; 0 = off")
    ap.add_argument("--n_shared", type=int, default=4,
                    help="shared (student) frames per pair: 4 = 16:4 protocol, 8 = 16:8 "
                         "(unsaturates the student's local targets on strong backbones)")
    ap.add_argument("--n_train", type=int, default=10,
                    help="train pairs per scene (protocol default 10; use 5 for tiny-N "
                         "scenes to avoid pair-overlap overfitting)")
    ap.add_argument("--teacher_N", type=int, default=0,
                    help="teacher window override (0 = protocol default 16/8): "
                         "e.g. 8 for 8:4/8:2, 32 for 32:8 ratio variants")
    ap.add_argument("--eval_max_frames", type=int, default=0,
                    help="cap eval frames (0 = protocol frames as-is)")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--early_stop", action="store_true",
                    help="stop training when tail-10 loss mean improves <2%% over "
                         "prev-10 (min 30 steps) — avoids over-training on saturated scenes")
    ap.add_argument("--swanlab", action="store_true",
                    help="log every step (loss components, lr, grad_norm, pair_idx) "
                         "to swanlab cloud; one experiment per scene, named {scene}_{arm}")
    ap.add_argument("--swanlab_suffix", default="",
                    help="suffix for swanlab experiment names, e.g. _mv13 to mark "
                         "the multi-view(13-39)-LoRA protocol runs")
    ap.add_argument("--v2_probe", action="store_true",
                    help="protocol v2: GT-free probe on the protocol's probe pairs "
                         "(kept teacher caches, 2 fixed masks per pair, robust "
                         "losses vs frozen teacher) every --v2_probe_every updates "
                         "plus step 0; appended to <output_root>/probe_trace/<scene>.jsonl")
    ap.add_argument("--v2_probe_every", type=int, default=10,
                    help="probe evaluation cadence in optimizer updates (default 10)")
    ap.add_argument("--v2_ckpt", action="store_true",
                    help="protocol v2: save LoRA at step 0 and every probe-cadence "
                         "step to <output_root>/ckpts/<scene>/v2/step{N}_lora.pt; "
                         "hard-fails if the artifact is missing on disk")
    ap.add_argument("--v2_rel_weight", type=float, default=0.0,
                    help="protocol v2: append rel_weight * (rot_edges_huber.loss + "
                         "tdir_cos_loss.loss) to the arm loss, teacher side "
                         "detached (replaces the old loss_pose_rel path, which "
                         "stays available via --arm rkdc1hr)")
    ap.add_argument("--v2_grad_cap", action="store_true",
                    help="protocol v2: separate backward of base vs v2-rel "
                         "gradients; ||g_R|| capped at 4*median(||g_R|| over the "
                         "first 10 updates) (requires --v2_rel_weight > 0)")
    ap.add_argument("--v2_couple_fix", action="store_true",
                    help="protocol v2: couple term uses the teacher-derived valid "
                         "mask on BOTH sides (losses.loss_couple_centers "
                         "semantics), skipping and logging on degenerate spread / "
                         "empty valid mask")
    ap.add_argument("--v2_rel_tau_gate", type=float, default=0.55,
                    help="dense-video rel gate: scenes with tau > this skip the "
                         "v2 rel branch (0 = off)")
    ap.add_argument("--v2_qfeat_off", action="store_true",
                    help="protocol v2: do NOT apply q_feat to the feature loss "
                         "(final config — geometry-side reliability kept; "
                         "feature downweighting cost VGGT F1)")
    ap.add_argument("--v2_baselines_json", default="workspace/protocol_v2/baselines.json",
                    help="protocol v2: baselines.json for the scene-level "
                         "baseline-vs-TTA swanlab summary (da3.<dataset>.baseline.<scene>)")
    ap.add_argument("--v2_ab_manifest", default=None,
                    help="protocol v2 layer 1+2: per-scene AB manifest — a directory "
                         "(<dir>/<dataset>/<scene>.json), a template with {scene}/"
                         "{ds}/{dataset}, or a plain file. When set, the scene "
                         "protocol comes from the manifest (NO on-the-fly "
                         "sampling); train pairs must carry teacher_frames_B "
                         "(context B). Enables dual teacher caches + frozen "
                         "reliability weights (q_feat/q_rot/q_tdir/geo_w).")
    ap.add_argument("--v2_rel_gate_deg", type=float, default=30.0,
                    help="protocol v2: scene-level rel gate — before training, run "
                         "UNMASKED B=0-student forwards on the probe pairs and "
                         "median the per-edge teacher-student relative-rotation "
                         "angle; above this threshold the v2 rel branch is "
                         "disabled for the scene (0 = gate off)")
    ap.add_argument("--v2_rot_weight", type=float, default=None,
                    help="protocol v2: weight of the rotation branch (rot_edges_huber); "
                         "default None -> fall back to --v2_rel_weight")
    ap.add_argument("--v2_tdir_weight", type=float, default=None,
                    help="protocol v2: weight of the translation-direction branch "
                         "(tdir_cos_loss); default None -> fall back to --v2_rel_weight")
    ap.add_argument("--v2_apply_selection", action="store_true",
                    help="protocol v2: after training, replay controller.select on "
                         "the probe trace and, when the selected step != last step, "
                         "load that checkpoint (reset_lora_ for step 0) and re-run "
                         "evaluate_scene into summary eval_selected (requires "
                         "--v2_probe and --v2_ckpt)")
    args = ap.parse_args()
    if args.v2_grad_cap and args.v2_rel_weight <= 0 \
            and (args.v2_rot_weight or 0) <= 0 and (args.v2_tdir_weight or 0) <= 0:
        ap.error("--v2_grad_cap requires --v2_rel_weight > 0 or --v2_rot_weight / "
                 "--v2_tdir_weight > 0")
    if args.v2_probe_every <= 0:
        ap.error("--v2_probe_every must be > 0")
    if args.v2_apply_selection and not (args.v2_probe and args.v2_ckpt):
        ap.error("--v2_apply_selection requires --v2_probe and --v2_ckpt")

    device = "cuda"
    fg_common.set_dataset(args.dataset)
    os.makedirs(args.output_root, exist_ok=True)
    trace_path = os.path.join(args.output_root, "training_trace.csv")
    summary_path = os.path.join(args.output_root, "smoke_summary.json")
    dataset_obj = make_dataset(args.dataset)

    print(f"Loading teacher + student: {args.model_name}")
    teacher = P.create_teacher(args.model_name)  # CPU; moved to GPU per phase
    student = P.create_student(args.model_name,
                               train_camera_token=not args.freeze_camera_token)

    trace_rows: list = []
    summary = {"model": args.model_name, "args": vars(args), "scenes": {}}

    for scene in args.scenes:
        t0 = time.time()
        scene_data = get_scene_data(scene)
        image_files = list(scene_data.image_files)
        ratio_mix = None
        if args.ratio_mix:
            ratio_mix = []
            for tok in args.ratio_mix.split(","):
                t_, s_ = tok.strip().split(":")
                ratio_mix.append((int(t_), int(s_)))
        if args.v2_ab_manifest:
            mpath = P.resolve_ab_manifest(args.v2_ab_manifest, args.dataset, scene)
            proto = P.load_ab_manifest(mpath, args.dataset, scene, image_files)
            print(f"[{scene}] protocol from AB manifest: {mpath} "
                  f"(train_pairs={len(proto['train_pairs'])}, "
                  f"probe_pairs={len(proto['probe_pairs'])}, dual-context)")
        else:
            proto = P.build_scene_protocol(image_files, scene, dataset=args.dataset,
                                           n_train=args.n_train,
                                           n_shared=args.n_shared,
                                           teacher_N=(args.teacher_N or None),
                                           ratio_mix=ratio_mix,
                                           selfevo=args.selfevo,
                                           combo=args.combo,
                                           combo_se_frac=args.combo_se_frac,
                                           sparse_overlap=args.sparse_overlap,
                                           force_dense=args.force_dense)
        _tau = proto.get("tau")
        tau_s = f"{_tau:.3f}" if isinstance(_tau, (int, float)) else str(_tau)
        print(f"[{scene}] N={proto['N']} tau={tau_s} "
              f"teacher_N={proto.get('teacher_N', '-')} strategy={proto.get('strategy', '-')} "
              f"n_shared={proto.get('n_shared', '-')} eval_frames={len(proto['eval_frames'])}")

        run = None
        if args.swanlab:
            try:
                import swanlab
                run = swanlab.init(
                    project="free-geometry-tta",
                    experiment_name=f"{scene}_{args.arm}{args.swanlab_suffix}",
                    description=f"{args.dataset} | {args.arm} | "
                                f"loss_all_pos={args.loss_all_pos} | seed={args.seed}",
                    config={k: v for k, v in vars(args).items()
                            if isinstance(v, (int, float, str, bool)) or v is None},
                )
            except Exception as e:  # logging must never kill training
                print(f"[{scene}] WARNING: swanlab unavailable ({e}); continuing without it")
                run = None

        pair_visits = {}

        v2 = None
        if (args.v2_probe or args.v2_ckpt or args.v2_rel_weight > 0
                or args.v2_grad_cap or args.v2_couple_fix or args.v2_ab_manifest):
            v2 = P.V2Config(
                probe=args.v2_probe, probe_every=args.v2_probe_every,
                ckpt=args.v2_ckpt, rel_weight=args.v2_rel_weight,
                grad_cap=args.v2_grad_cap, couple_fix=args.v2_couple_fix,
                qfeat_off=args.v2_qfeat_off,
                rel_tau_gate=args.v2_rel_tau_gate,
                run_dir=args.output_root,
                ckpt_dir=os.path.join(args.output_root, "ckpts"),
                ab=args.v2_ab_manifest is not None,
                rel_gate_deg=args.v2_rel_gate_deg,
                rot_weight=args.v2_rot_weight,
                tdir_weight=args.v2_tdir_weight)

        def _on_step(row, _run=run):
            if _run is None:
                return
            m = {}
            for k in ("loss", "lr", "grad_norm", "pair_idx", "epoch",
                      "maskdistill", "rkd_sh_d", "rkd_sh_a", "couple",
                      "rel_rot", "rel_tdir", "ctk", "peak_mem_mib",
                      "v2_rel_rot", "v2_rel_tdir",
                      "g_base_norm", "g_R_norm", "v2_C_R",
                      "g_rot_norm", "g_tdir_norm",
                      "rel_w_eff", "w_rot_eff", "w_tdir_eff",
                      "rel_gate", "rot_deg_unmasked_median",
                      "q_feat_mean", "q_rot_mean", "geo_w"):
                v = row.get(k)
                if isinstance(v, (int, float)) and v == v:
                    m[k] = v
            if "rel_gate" in m:
                m["gate/rel_gate"] = m.pop("rel_gate")
                m["gate/rot_deg_unmasked_median"] = m.pop("rot_deg_unmasked_median")
            try:
                _run.log(m, step=int(row["step"]))
            except Exception as e:
                print(f"[warn] swanlab log failed: {e}")
            # per-pair curves: x-axis = visit index of THIS pair, so the 10
            # interleaved tasks no longer blur into one chaotic global line
            pi = int(row["pair_idx"])
            pair_visits[pi] = pair_visits.get(pi, 0) + 1
            pm = {}
            for k in ("loss", "maskdistill", "rkd_sh_d", "rkd_sh_a", "couple",
                      "rel_rot", "rel_tdir", "grad_norm"):
                v = row.get(k)
                if isinstance(v, (int, float)) and v == v:
                    pm[f"pair{pi}/{k}"] = v
            try:
                # normalized per-pair components (v2 reporting): pair/pN/{total,
                # feature, rkd, couple, rel} — rel only when the arm has a rel term
                normed = {"total": row.get("loss"), "feature": row.get("maskdistill")}
                if row.get("rkd_sh_d") is not None or row.get("rkd_sh_a") is not None:
                    normed["rkd"] = (row.get("rkd_sh_d") or 0.0) + (row.get("rkd_sh_a") or 0.0)
                if row.get("couple") is not None:
                    normed["couple"] = row.get("couple")
                if row.get("rel_rot") is not None:  # old loss_pose_rel path
                    normed["rel"] = row.get("rel_rot") + row.get("rel_tdir")
                elif row.get("v2_rel_rot") is not None:  # v2 robust rel branch
                    normed["rel"] = row.get("v2_rel_rot") + row.get("v2_rel_tdir")
                if row.get("q_feat_mean") is not None:  # AB reliability (frozen)
                    normed["q_feat"] = row.get("q_feat_mean")
                    normed["q_rot"] = row.get("q_rot_mean")
                    normed["geo_w"] = row.get("geo_w")
                for name, v in normed.items():
                    if isinstance(v, (int, float)) and v == v:
                        pm[f"pair/p{pi}/{name}"] = v
            except Exception as e:
                print(f"[warn] per-pair swanlab logging failed: {e}")
            try:
                _run.log(pm, step=pair_visits[pi])
            except Exception as e:
                print(f"[warn] swanlab per-pair log failed: {e}")

        def _on_probe(rec, _run=run):
            """probe curves at the REAL step: per (pair, mask) components + means"""
            if _run is None:
                return
            try:
                comps = ("feature", "rot_deg", "rkd", "couple")
                m = {}
                for r in rec["records"]:
                    j = str(r["pair_id"]).replace("probe", "")
                    k = int(r["mask_id"])
                    for c in comps:
                        v = r["components"].get(c)
                        if isinstance(v, (int, float)) and v == v:
                            m[f"probe/pair{j}/mask{k}/{c}"] = v
                    m[f"probe/pair{j}/mask{k}/total"] = r["total"]
                n = len(rec["records"])
                for c in comps:
                    m[f"probe/mean/{c}"] = sum(r["components"][c] for r in rec["records"]) / n
                m["probe/mean/total"] = sum(r["total"] for r in rec["records"]) / n
                _run.log(m, step=int(rec["step"]))
            except Exception as e:
                print(f"[warn] probe swanlab logging failed: {e}")

        pose_w = 0.0 if args.arm == "pw0" else args.pose_weight
        stats = P.train_scene_c2m(
            teacher, student, scene, image_files, proto,
            device=device, steps=args.steps, epochs=args.epochs,
            lr=args.lr, seed=args.seed, pose_weight=pose_w,
            arm=args.arm, mask_ratio=args.mask_ratio, trace_rows=trace_rows,
            mask_mode=args.mask_mode, mask_layer=args.mask_layer,
            ctk_weight=args.ctk_weight, rel_weight=args.rel_weight,
            two_stage=args.two_stage,
            early_stop=args.early_stop,
            mask_loss_positions=not args.loss_all_pos,
            abs_pose_w=args.abs_pose_w,
            on_step=_on_step, on_probe=_on_probe if args.swanlab else None, v2=v2)

        if run is not None and v2 is not None and v2.probe:
            # offline selector replay on the probe trace this run just wrote —
            # shows which checkpoint the controller would pick (read-only)
            try:
                from free_geometry.tta_v2.controller import ControllerConfig, select
                trace_file = os.path.join(args.output_root, "probe_trace", f"{scene}.jsonl")
                with open(trace_file) as f:
                    trace = [json.loads(line) for line in f if line.strip()]
                sel = select(trace, ControllerConfig())
                run.log({"selector/selected_step": sel["selected_step"],
                         "selector/fell_back_to_baseline": sel["fell_back_to_baseline"],
                         "selector/improvement": sel["improvement"]})
                print(f"[{scene}] selector replay: selected_step={sel['selected_step']} "
                      f"fell_back={sel['fell_back_to_baseline']} "
                      f"improvement={sel['improvement']:.4f}")
            except Exception as e:
                print(f"[{scene}] WARNING: selector replay failed: {e}")

        if run is not None and isinstance(stats.get("q_summary"), dict):
            try:
                run.log({f"q/{k}": v for k, v in stats["q_summary"].items()})
            except Exception as e:
                print(f"[{scene}] WARNING: q-summary logging failed: {e}")

        ckpt_dir = os.path.join(args.output_root, "ckpts", scene)
        os.makedirs(ckpt_dir, exist_ok=True)
        student.save_lora_weights(os.path.join(ckpt_dir, "c2m_final_lora.pt"))

        summary["scenes"][scene] = {
            "protocol": {k: proto[k] for k in ("N", "tau", "teacher_N", "strategy")},
            "n_eval_frames": len(proto["eval_frames"]),
            "train": stats,
            "train_time_s": time.time() - t0,
        }
        # Keep the protocol on disk for reproducibility (frame lists included).
        with open(os.path.join(ckpt_dir, "protocol.json"), "w") as f:
            json.dump(proto, f, indent=2)

        if not args.skip_eval:
            # Eval with the just-trained adapter still active in memory (avoids
            # a PEFT same-name adapter reload). Teacher stays resident but idle.
            # ONE inference -> AUC + AbsRel + F1/CD out together (no backfill).
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            ev = evaluate_scene(student, scene_data, proto["eval_frames"],
                                max_frames=args.eval_max_frames,
                                scene=scene, dataset_obj=dataset_obj,
                                export_dir=os.path.join(args.output_root, "recon", scene))
            ev["eval_peak_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
            ev["eval_time_s"] = time.time() - t0
            summary["scenes"][scene]["eval"] = ev
            print(f"[{scene}] eval: auc03={ev['auc03']:.4f} "
                  f"fscore={ev.get('recon_fscore', float('nan')):.4f} "
                  f"cd={ev.get('recon_overall', float('nan')):.4f} "
                  f"abs_rel={ev.get('abs_rel', float('nan')):.4f} "
                  f"({ev['n_eval_frames']} frames, peak {ev['eval_peak_mem_mib']:.0f}MiB)")
            if run is not None:
                # scene-level baseline-vs-TTA summary (signed relative deltas)
                try:
                    base = load_scene_baseline(args.v2_baselines_json,
                                               args.dataset, scene)
                    if base is not None:
                        m = {}
                        b_auc = base.get("auc03")
                        if b_auc and ev.get("auc03") is not None:
                            m["eval/auc03"] = ev["auc03"]
                            m["eval/d_auc03_rel"] = (ev["auc03"] - b_auc) / b_auc
                        b_f = base.get("fscore")
                        if b_f and ev.get("recon_fscore") is not None:
                            m["eval/f1"] = ev["recon_fscore"]
                            m["eval/d_f1_rel"] = (ev["recon_fscore"] - b_f) / b_f
                        else:
                            b_o = base.get("overall")  # dtu/dtu64: chamfer distance
                            if b_o and ev.get("recon_overall") is not None:
                                m["eval/chamfer_overall"] = ev["recon_overall"]
                                m["eval/d_chamfer_overall_rel"] = (
                                    ev["recon_overall"] - b_o) / b_o
                        if m:
                            run.log(m)
                            print(f"[{scene}] baseline delta vs {args.dataset}/baseline: "
                                  + " ".join(f"{k}={v:+.4f}" for k, v in m.items()
                                             if k.startswith("eval/d_")))
                except Exception as e:
                    print(f"[{scene}] WARNING: baseline comparison failed: {e}")

        if args.v2_apply_selection and v2 is not None and v2.probe \
                and not args.skip_eval and "eval" in summary["scenes"][scene]:
            # execution chain: replay the selector, LOAD the selected checkpoint
            # (reset_lora_ == step-0 theta0), re-evaluate on the same frames.
            try:
                from free_geometry.tta_v2.controller import ControllerConfig, select
                trace_file = os.path.join(args.output_root, "probe_trace", f"{scene}.jsonl")
                with open(trace_file) as f:
                    trace = [json.loads(line) for line in f if line.strip()]
                sel = select(trace, ControllerConfig())
                sel_step, last_step = int(sel["selected_step"]), int(stats["steps"])
                applied = sel_step
                if sel_step != last_step:
                    if sel_step == 0:
                        P.reset_lora_(student)  # step-0 theta0 (B=0 LoRA)
                    else:
                        sel_ckpt = os.path.join(
                            args.output_root, "ckpts", scene, "v2",
                            f"step{sel_step}_lora.pt")
                        assert os.path.exists(sel_ckpt), \
                            f"selected ckpt missing: {sel_ckpt}"
                        student.load_lora_weights(sel_ckpt)
                    t0 = time.time()
                    torch.cuda.reset_peak_memory_stats()
                    ev_sel = evaluate_scene(
                        student, scene_data, proto["eval_frames"],
                        max_frames=args.eval_max_frames, scene=scene,
                        dataset_obj=dataset_obj,
                        export_dir=os.path.join(args.output_root, "recon_selected", scene))
                    ev_sel["eval_time_s"] = time.time() - t0
                    ev_sel["actual_loaded_step"] = applied
                    summary["scenes"][scene]["eval_selected"] = ev_sel
                    print(f"[{scene}] eval_selected (loaded step{applied}): "
                          f"auc03={ev_sel['auc03']:.4f} "
                          f"fscore={ev_sel.get('recon_fscore', float('nan')):.4f}")
                else:
                    print(f"[{scene}] selector picked the last step ({sel_step}); "
                          f"no reload/re-eval needed")
                if run is not None:
                    m = {"selector/applied_step": applied,
                         "selector/fell_back_to_baseline": sel["fell_back_to_baseline"]}
                    if "eval_selected" in summary["scenes"][scene]:
                        evs = summary["scenes"][scene]["eval_selected"]
                        m["eval_selected/auc03"] = evs["auc03"]
                        if evs.get("recon_fscore") is not None:
                            m["eval_selected/f1"] = evs["recon_fscore"]
                        elif evs.get("recon_overall") is not None:
                            m["eval_selected/chamfer_overall"] = evs["recon_overall"]
                    run.log(m)
            except Exception as e:
                # never mask the main eval result
                print(f"[{scene}] WARNING: v2_apply_selection failed: {e}")
            # note: no explicit restore — every scene starts with reset_lora_(),
            # and the on-disk final ckpt was saved before eval.

        if run is not None:
            if not args.skip_eval and "eval" in summary["scenes"][scene]:
                run.log({"eval_auc03": summary["scenes"][scene]["eval"]["auc03"],
                         "eval_fscore": summary["scenes"][scene]["eval"].get("recon_fscore")})
            run.finish()

    del teacher
    torch.cuda.empty_cache()

    if trace_rows:
        keys = ["scene", "step", "epoch", "pair_idx", "loss", "lr", "grad_norm",
                "peak_mem_mib", "maskdistill", "rel", "rel_rot", "rel_tdir",
                "rkd_sh_d", "rkd_sh_a", "couple", "ctk", "mask_ratio",
                "v2_rel_rot", "v2_rel_tdir", "couple_skipped",
                "g_base_norm", "g_R_norm", "v2_C_R",
                "g_rot_norm", "g_tdir_norm",
                "rel_w_eff", "w_rot_eff", "w_tdir_eff",
                "rel_gate", "rot_deg_unmasked_median",
                "q_feat_mean", "q_rot_mean", "geo_w"]
        with open(trace_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(trace_rows)
        print(f"Trace written to {trace_path}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")
    print("DONE")


if __name__ == "__main__":
    main()
