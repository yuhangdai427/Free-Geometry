#!/usr/bin/env python3
"""Gradient-accumulation trainer for the pw0 (pure maskdistill, allpos) arm
(2026-09-21, courtyard A/B). K pairs per optimizer update — mathematically
identical to batching K pairs, at the SAME peak memory (one pair per forward;
only the .grad buffers persist, ~85MB).

Faithful to the main loop's pieces: same protocol pairs, same seeded mask
draws, same optimizer/scheduler (warmup 15% + cosine over n_updates), frozen
camera token, mv-LoRA 13-39. Saves c2m_final_lora.pt for the standard
evaluate_scene (cam_dec) afterwards.

Usage: python scripts/train_pw0_accum.py --scene courtyard --updates 10 --accum 5
"""
import argparse, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "diagnostics", "free_geometry"))

from depth_anything_3.test_time_adaption import protocol_v1 as P
import common as fg_common
from train_da3_protocol import make_dataset, get_scene_data, evaluate_scene


def main():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eth3d")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--updates", type=int, default=10, help="optimizer updates")
    ap.add_argument("--accum", type=int, default=5, help="pairs per update")
    ap.add_argument("--loss_form", default="huber", choices=["huber", "half_mse"],
                    help="pointwise form on the NORMALIZED space: huber=SmoothL1(beta=1) "
                         "(deployed); half_mse=0.5*d^2 (tail unbounded)")
    ap.add_argument("--half_mode", default="joint",
                    choices=["joint", "split50", "local", "global", "g2", "g3",
                             "cosw1", "cos_split", "cos_only", "mse_only", "half_resplit",
                             "vggt", "vggt_mse"],
                    help="which part of the DEPLOYED joint-LN(3072) normalized tensor "
                         "enters the loss: joint=whole 3072 (deployed); split50=huber "
                         "per half of the ALREADY-normalized tensor, 50/50 weights — "
                         "mathematically IDENTICAL to joint (channel-mean of the concat "
                         "= average of per-half means); local/global=only that half; "
                         "g2=(L_l + 2*L_g)/3 — explicit 2x relative weight on the "
                         "global half's residual (gradient share ~21%->~35%); "
                         "half_resplit=the earlier WRONG variant (re-LN per half)")
    ap.add_argument("--teacher_N", type=int, default=0,
                    help="0 = N-dispatch auto (N>=64 -> 16:4 else 8:4); explicit "
                         "value forces that teacher size for ALL scenes")
    ap.add_argument("--space", default="head", choices=["head", "enc"],
                    help="loss space: head = head_norm(tap feats) (deployed "
                         "readout); enc = RAW encoder output tokens at the same "
                         "TAP_LAYERS, no head_norm — loss form unchanged")
    ap.add_argument("--halves", action="store_true",
                    help="measure per-half (local/global stream) component "
                         "grads; training loss unchanged")
    ap.add_argument("--n_train", type=int, default=10,
                    help="number of protocol train pairs (ETH3D default 10)")
    ap.add_argument("--vggt_sync", action="store_true",
                    help="conform to the deployed VGGT protocol: pairs from the "
                         "final_protocol manifest, VGGT image geometry (504 "
                         "longest) with DA3 ImageNet norm, identical mask draws "
                         "(same seed scheme + patch grid), deployed B5 loss + "
                         "rel-pose (rot + tdir); records 4 component grads")
    ap.add_argument("--manifest", default="artifacts/diagnostics/final_protocol/eth3d/scene_manifest.json")
    ap.add_argument("--keep_pairs", default=None,
                    help="comma-separated manifest pair indices to KEEP "
                         "(e.g. '5,6,8'); vggt_sync only, everything else "
                         "unchanged")
    ap.add_argument("--resample", type=int, default=0,
                    help="screen N candidate pairs (16:4, protocol-generated): "
                         "measure step-0 rel-rot grad per pair at zero-LoRA, "
                         "keep those with g_rot < grot_max (cap 10, min 5 by "
                         "lowest), and set updates = 10 * kept")
    ap.add_argument("--grot_max", type=float, default=10.0,
                    help="screening upper bound: drop pairs with step-0 "
                         "g_rot above this (spikes)")
    ap.add_argument("--grot_min", type=float, default=0.02,
                    help="screening lower bound: drop pairs with step-0 "
                         "g_rot below this (no rel signal to teach)")
    ap.add_argument("--clip", type=float, default=None,
                    help="None = deployed P.CLIP (1.0); 0 = NO clipping; "
                         ">0 = explicit value")
    ap.add_argument("--lr", type=float, default=P.LR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.half_mode in ("vggt", "vggt_mse"):
        args.space = "enc"  # VGGT paradigm runs in the raw-token space
        print(f"[mode] {args.half_mode}: forcing --space enc (raw encoder tokens)")
    out = args.out or f"workspace/accum_{args.scene}_u{args.updates}k{args.accum}"
    os.makedirs(out + "/ckpts/" + args.scene, exist_ok=True)
    device = "cuda"

    fg_common.set_dataset(args.dataset)
    ds_obj = make_dataset(args.dataset)
    sd = get_scene_data(args.scene)
    files = list(sd.image_files)
    if args.vggt_sync:
        import json as _json
        man = _json.load(open(args.manifest))
        sc = man["scenes"][args.scene]
        train_pairs = sc["train_pairs"]
        proto = {"eval_frames": sc["eval32_frames"], "teacher_N": "manifest"}
        if args.keep_pairs:
            keep = [int(x) for x in args.keep_pairs.split(",")]
            train_pairs = [train_pairs[i] for i in keep]
            print(f"[{args.scene}] VGGT-SYNC: KEEPING pairs {keep} "
                  f"(filtered from manifest)")
        if args.resample > 0:
            # frame-count dispatch rule (user, 2026-09-21): <40 frames -> 8:4,
            # >=40 -> 16:4; an explicit --teacher_N overrides for A/B tests
            _tn = args.teacher_N if args.teacher_N > 0 else (
                16 if len(files) >= 40 else 8)
            proto2 = P.build_scene_protocol(
                files, args.scene, dataset=args.dataset,
                n_train=args.resample, n_shared=4, teacher_N=_tn)
            train_pairs = proto2["train_pairs"]
            print(f"[{args.scene}] RESAMPLE: {args.resample} fresh {_tn}:4 "
                  f"candidate pairs, N={len(files)} frames "
                  f"(screening by step-0 g_rot)")
        print(f"[{args.scene}] VGGT-SYNC: {len(train_pairs)} manifest pairs, "
              f"eval_frames={len(sc['eval32_frames'])}")
    else:
        proto = P.build_scene_protocol(files, args.scene, dataset=args.dataset,
                                       n_train=args.n_train, n_shared=4,
                                       teacher_N=(args.teacher_N if args.teacher_N > 0 else None))
        train_pairs = proto["train_pairs"]
        print(f"[{args.scene}] N={len(files)} tN={proto['teacher_N']} "
              f"updates={args.updates} accum={args.accum}")
    for ip, pair in enumerate(train_pairs):
        print(f"[{args.scene}] train pair {ip}: teacher={pair['teacher_frames']} "
              f"student={pair['student_frames']}", flush=True)

    teacher = P.create_teacher("model_weights/DA3-GIANT-1.1")
    student = P.create_student("model_weights/DA3-GIANT-1.1",
                               train_camera_token=False)  # frozen (best cell)
    teacher.to(device)
    # teacher caches (identical to train_scene_c2m: CPU, streamed per visit)
    caches, imgs4 = [], []
    if args.vggt_sync:
        from vggt.vggt.test_time_adaption.test3r_utils import load_images_for_vggt
        _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        for pair in train_pairs:
            # VGGT geometry (identical pixels/grid as the VGGT run), DA3 norm
            ims01 = load_images_for_vggt(
                [files[i] for i in pair["teacher_frames"]]).cpu()
            im8 = ((ims01 - _MEAN) / _STD).unsqueeze(0).to(device)
            ph, pw = im8.shape[-2] // P.PATCH_SIZE, im8.shape[-1] // P.PATCH_SIZE
            slots = [pair["teacher_frames"].index(f) for f in pair["student_frames"]]
            caches.append(P.cache_teacher_pair(teacher, im8, slots, (ph, pw)))
            imgs4.append(im8[0, slots].cpu().contiguous())
            del im8, ims01
    else:
        for pair in train_pairs:
            im8 = P.load_images_da3([files[i] for i in pair["teacher_frames"]])
            im8 = im8.unsqueeze(0).to(device)
            ph, pw = im8.shape[-2] // P.PATCH_SIZE, im8.shape[-1] // P.PATCH_SIZE
            slots = list(range(0, 2 * len(pair["student_frames"]), 2))
            caches.append(P.cache_teacher_pair(teacher, im8, slots, (ph, pw)))
            imgs4.append(im8[0, slots].cpu().contiguous())
            del im8
    teacher.to("cpu"); torch.cuda.empty_cache()
    patch_hw = caches[0]["patch_hw"]
    n_train = len(caches)

    if args.resample > 0:
        # screen candidates by step-0 rel-rot gradient at zero-LoRA
        student.to(device)
        _params = student.get_trainable_params()
        _, _, H, W = imgs4[0].shape[-4:]
        _phw = (H // P.PATCH_SIZE, W // P.PATCH_SIZE)
        scores = []
        for i in range(n_train):
            images4 = imgs4[i].unsqueeze(0).to(device)
            gen = torch.Generator(device=device).manual_seed(
                P.stable_seed("mask", args.scene, 0, i, 0))
            images4_in, _pm = P.mask_image_blocks(images4, 0.5, _phw, gen)
            _, ext_w2c, _ = P.student_forward_c2m(student, images4_in)
            ext_s = ext_w2c[0].float()
            ext_t = caches[i]["ext4"].to(device)
            R_s, t_s = ext_s[..., :3, :3], ext_s[..., :3, 3]
            R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]
            rot_acc, np_ = 0.0, 0
            for a in range(4):
                for b in range(a + 1, 4):
                    Rr_s = R_s[a] @ R_s[b].transpose(-1, -2)
                    Rr_t = R_t[a] @ R_t[b].transpose(-1, -2)
                    rot_acc = rot_acc + ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1))
                    np_ += 1
            rel_rot = rot_acc / np_
            g = torch.autograd.grad(rel_rot, _params, allow_unused=True)
            gf = torch.cat([t.reshape(-1) for t in g if t is not None])
            scores.append(float(gf.norm()))
            del g, gf, images4, images4_in
        order = sorted(range(n_train), key=lambda i: scores[i])
        band = [i for i in order if args.grot_min <= scores[i] <= args.grot_max]
        if len(band) > 10:
            # keep the 10 MOST informative (largest g_rot within the band)
            band = sorted(band, key=lambda i: -scores[i])[:10]
        if not band:
            # fallback: 5 nearest the band center
            ctr = (args.grot_min + args.grot_max) / 2
            band = sorted(range(n_train), key=lambda i: abs(scores[i] - ctr))[:5]
        keep = band
        for i in range(n_train):
            print(f"[{args.scene}] screen pair {i}: g_rot0={scores[i]:.3f}"
                  f"{'  KEEP' if i in keep else ''}", flush=True)
        caches = [caches[i] for i in keep]
        imgs4 = [imgs4[i] for i in keep]
        n_train = len(keep)
        args.updates = 10 * n_train
        print(f"[{args.scene}] RESAMPLE kept {n_train} pairs -> "
              f"updates={args.updates}", flush=True)
        student.to("cpu"); torch.cuda.empty_cache()

    torch.manual_seed(P.stable_seed("lora_init", args.scene, 0))
    P.reset_lora_(student)
    student.to(device)
    head_norm = student.da3.model.head.norm
    params = student.get_trainable_params()
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=P.WD)
    n_steps = args.updates
    warm = max(1, round(n_steps * P.WARMUP_RATIO))
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warm),
        CosineAnnealingLR(optimizer, T_max=n_steps - warm, eta_min=1e-8)], [warm])

    # deterministic visit order: per-epoch seeded permutation (same scheme as main loop)
    def epoch_order(epoch):
        g = torch.Generator().manual_seed(
            P.stable_seed("train_order", args.scene, epoch, 0))
        return [i for i in torch.randperm(n_train, generator=g).tolist()]

    torch.cuda.reset_peak_memory_stats()
    visit = 0
    grad_rows = []
    half_rows = []
    rel_rows = []
    t0 = time.time()
    for upd in range(args.updates):
        epoch = (upd * args.accum) // n_train
        for k in range(args.accum):
            # walk the seeded order; advance epoch bookkeeping when exhausted
            pos_in_epoch = ((upd * args.accum) + k) % n_train
            pi = epoch_order(epoch)[pos_in_epoch] if pos_in_epoch < n_train else \
                 epoch_order(epoch + 1)[0]
            pi_last = pi  # which train pair this visit/update used (order is shuffled)
            images4 = imgs4[pi].unsqueeze(0).to(device)
            gen = torch.Generator(device=device).manual_seed(
                P.stable_seed("mask", args.scene, epoch, pi, 0))
            images4_in, pmask = P.mask_image_blocks(images4, 0.5, patch_hw, gen)
            ones = torch.ones_like(pmask)  # allpos: supervise every position
            torch.manual_seed(P.stable_seed("cf_rng", args.scene, epoch, pi, 0))
            tap_feats_s, ext_w2c, _ = P.student_forward_c2m(student, images4_in)
            ext_s = ext_w2c[0].float() if torch.is_tensor(ext_w2c) else ext_w2c[0]
            cache = {"feats": {l: t.to(device) for l, t in caches[pi]["feats"].items()},
                     "conf4": caches[pi]["conf4"].to(device), "patch_hw": patch_hw}
            loss_d = loss_c = None  # set in the main branch below
            half_loss = None
            if args.half_mode == "half_resplit":
                # the earlier WRONG variant (kept for the record): re-LN per half
                # with affine slices — CHANGES the space; do not use for the
                # balanced-weighting question
                w = P.teacher_patch_conf(cache["conf4"], patch_hw)
                w = w / w.mean().clamp_min(1e-8)
                C = tap_feats_s[P.TAP_LAYERS[0]].shape[-1] // 2
                gamma, beta = head_norm.weight, head_norm.bias
                total = 0.0
                for layer in P.TAP_LAYERS:
                    hs_raw = tap_feats_s[layer].float()
                    ht_raw = cache["feats"][layer].float()
                    hs_l = torch.nn.functional.layer_norm(hs_raw[..., :C], (C,), gamma[:C], beta[:C])
                    hs_g = torch.nn.functional.layer_norm(hs_raw[..., C:], (C,), gamma[C:], beta[C:])
                    ht_l = torch.nn.functional.layer_norm(ht_raw[..., :C], (C,), gamma[:C], beta[:C])
                    ht_g = torch.nn.functional.layer_norm(ht_raw[..., C:], (C,), gamma[C:], beta[C:])
                    hub_l = torch.nn.functional.smooth_l1_loss(hs_l, ht_l, beta=1.0, reduction="none").mean(-1)
                    hub_g = torch.nn.functional.smooth_l1_loss(hs_g, ht_g, beta=1.0, reduction="none").mean(-1)
                    huber = 0.5 * ((hub_l * w).mean() + (hub_g * w).mean())
                    cos_t = torch.nn.functional.cosine_similarity(
                        torch.cat([hs_l, hs_g], dim=-1), torch.cat([ht_l, ht_g], dim=-1), dim=-1)
                    total = total + huber + 2.0 * (1.0 - (cos_t * w).mean())
                loss = total / len(P.TAP_LAYERS)
            else:
                # ALL other modes: deploy the joint head_norm(3072) ONCE
                # (space identical to production), then select what enters the
                # loss. The pointwise (d) and cosine (c) parts accumulate
                # SEPARATELY so each update can report their individual
                # gradient norms / share / direction.
                w = P.teacher_patch_conf(cache["conf4"], patch_hw)
                w = w / w.mean().clamp_min(1e-8)
                C = tap_feats_s[P.TAP_LAYERS[0]].shape[-1] // 2
                acc_d, acc_c = 0.0, 0.0
                hacc = [0.0, 0.0, 0.0, 0.0]  # d_loc, c_loc, d_glob, c_glob
                for layer in P.TAP_LAYERS:
                    if args.space == "enc":
                        # encoder-output space: raw tap tokens, NO head_norm
                        hs = tap_feats_s[layer].float()
                        ht = cache["feats"][layer].float()
                    else:
                        hs = head_norm(tap_feats_s[layer].float())
                        ht = head_norm(cache["feats"][layer].float())
                    if args.halves:
                        C2 = hs.shape[-1] // 2
                        hacc[0] = hacc[0] + (0.5 * (hs[..., :C2] - ht[..., :C2]) ** 2).mean(dim=-1).mean()
                        hacc[1] = hacc[1] + (1.0 - torch.nn.functional.cosine_similarity(
                            hs[..., :C2], ht[..., :C2], dim=-1).mean())
                        hacc[2] = hacc[2] + (0.5 * (hs[..., C2:] - ht[..., C2:]) ** 2).mean(dim=-1).mean()
                        hacc[3] = hacc[3] + (1.0 - torch.nn.functional.cosine_similarity(
                            hs[..., C2:], ht[..., C2:], dim=-1).mean())

                    def _h(a, b):  # pointwise form, per-patch channel-mean, conf-weighted
                        if args.loss_form == "half_mse":
                            t = (0.5 * (a - b) ** 2).mean(dim=-1)
                        else:
                            t = torch.nn.functional.smooth_l1_loss(a, b, beta=1.0, reduction="none").mean(dim=-1)
                        return (t * w).mean()

                    if args.half_mode in ("vggt", "vggt_mse"):
                        # VGGT paradigm on RAW encoder tokens: SmoothL1(beta=1)
                        # (vggt) or half-MSE (vggt_mse) FULL-element mean +
                        # 2*(1-cos) on L2-normalized tokens, per layer.
                        # Identical for |d|<=1; MSE has the unbounded tail.
                        if args.half_mode == "vggt":
                            dist = torch.nn.functional.smooth_l1_loss(hs, ht, beta=1.0)
                        else:
                            dist = (0.5 * (hs - ht) ** 2).mean()
                        cosv = (torch.nn.functional.normalize(hs, dim=-1)
                                * torch.nn.functional.normalize(ht, dim=-1)).sum(dim=-1).mean()
                        acc_d = acc_d + dist
                        acc_c = acc_c + 2.0 * (1.0 - cosv)
                        continue
                    if args.half_mode == "joint":
                        huber = _h(hs, ht)
                        cos_t = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
                    elif args.half_mode == "split50":
                        # balanced WEIGHTS on the untouched normalized tensor
                        # (mathematically identical to joint — kept as an
                        # empirical check of that equivalence)
                        huber = 0.5 * (_h(hs[..., :C], ht[..., :C]) + _h(hs[..., C:], ht[..., C:]))
                        cos_t = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
                    elif args.half_mode == "cos_only":
                        # decomposition control: ONLY the dominant 2(1-cos) term
                        ct = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
                        acc_c = acc_c + 2.0 * (1.0 - (ct * w).mean())
                        continue
                    elif args.half_mode == "mse_only":
                        # decomposition control: ONLY the pointwise distance term
                        acc_d = acc_d + _h(hs, ht)
                        continue
                    elif args.half_mode == "cosw1":
                        # cos weight 1 (deployed is 2): L = h + 1*(1-cos)
                        huber = _h(hs, ht)
                        cos_t = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
                        acc_d = acc_d + huber
                        acc_c = acc_c + 1.0 * (1.0 - (cos_t * w).mean())
                        continue
                    elif args.half_mode == "cos_split":
                        # split the DOMINANT term (cos carries ~94.5% of the
                        # gradient): per-half cosines, 50/50 — the first
                        # reweighting that actually touches the main signal
                        huber = _h(hs, ht)
                        cl = torch.nn.functional.cosine_similarity(hs[..., :C], ht[..., :C], dim=-1)
                        cg = torch.nn.functional.cosine_similarity(hs[..., C:], ht[..., C:], dim=-1)
                        cos_part = 2.0 * (0.5 * (1.0 - (cl * w).mean())
                                          + 0.5 * (1.0 - (cg * w).mean()))
                        acc_d = acc_d + huber
                        acc_c = acc_c + cos_part
                        continue
                    elif args.half_mode == "g2":
                        # explicit dose: global half's residual penalty 2x
                        huber = (_h(hs[..., :C], ht[..., :C]) + 2.0 * _h(hs[..., C:], ht[..., C:])) / 3.0
                        cos_t = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
                    elif args.half_mode == "g3":
                        # global 3x: gradient share ~21% -> ~44% (roughly parity
                        # with local's 79%/3.76x magnitude dominance)
                        huber = (_h(hs[..., :C], ht[..., :C]) + 3.0 * _h(hs[..., C:], ht[..., C:])) / 4.0
                        cos_t = torch.nn.functional.cosine_similarity(hs, ht, dim=-1)
                    elif args.half_mode == "local":
                        huber = _h(hs[..., :C], ht[..., :C])
                        cos_t = torch.nn.functional.cosine_similarity(hs[..., :C], ht[..., :C], dim=-1)
                    else:  # global
                        huber = _h(hs[..., C:], ht[..., C:])
                        cos_t = torch.nn.functional.cosine_similarity(hs[..., C:], ht[..., C:], dim=-1)
                    acc_d = acc_d + huber
                    acc_c = acc_c + 2.0 * (1.0 - (cos_t * w).mean())
                loss_d = acc_d / len(P.TAP_LAYERS)
                loss_c = acc_c / len(P.TAP_LAYERS)
                loss = loss_d + loss_c
                if args.halves:
                    half_loss = [a / len(P.TAP_LAYERS) for a in hacc]
                else:
                    half_loss = None
            rel_rot = rel_tdir = None
            if args.vggt_sync:
                # deployed C2M_maskrel rel-pose term, same math AND same
                # CONVENTION as train_arms.loss_pose_rel: rel quantities on
                # c2w poses (VGGT consumes the decoder output directly; DA3's
                # model path inverts to w2c, so invert back here)
                from depth_anything_3.utils.geometry import affine_inverse
                ext_t = affine_inverse(caches[pi]["ext4"].to(device))  # w2c->c2w
                ext_s_c2w = affine_inverse(ext_s)                      # w2c->c2w
                R_s, t_s = ext_s_c2w[..., :3, :3], ext_s_c2w[..., :3, 3]
                R_t, t_t = ext_t[..., :3, :3], ext_t[..., :3, 3]
                S_ = R_s.shape[0]
                rot_acc, tdir_acc, np_ = 0.0, 0.0, 0
                for i in range(S_):
                    for j in range(i + 1, S_):
                        Rr_s = R_s[i] @ R_s[j].transpose(-1, -2)
                        Rr_t = R_t[i] @ R_t[j].transpose(-1, -2)
                        tr_s = t_s[i].unsqueeze(-1) - Rr_s @ t_s[j].unsqueeze(-1)
                        tr_t = t_t[i].unsqueeze(-1) - Rr_t @ t_t[j].unsqueeze(-1)
                        rot_acc = rot_acc + ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1))
                        tn_s = torch.nn.functional.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
                        tn_t = torch.nn.functional.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
                        tdir_acc = tdir_acc + (1.0 - (tn_s * tn_t).sum(-1))
                        np_ += 1
                rel_rot, rel_tdir = rot_acc / np_, tdir_acc / np_
                loss = loss + rel_rot + rel_tdir
            if loss is None:
                loss, _ = P.loss_maskdistill(head_norm, cache, tap_feats_s, ones)
                loss_d = loss_c = None
            comp = None
            if loss_d is not None and torch.is_tensor(loss_d) and loss_d.requires_grad \
                    and torch.is_tensor(loss_c) and loss_c.requires_grad:
                # per-component gradient norms/direction; autograd.grad does NOT
                # touch .grad, so the optimizer update below is unchanged
                gd = torch.autograd.grad(loss_d, params, retain_graph=True, allow_unused=True)
                gc = torch.autograd.grad(loss_c, params, retain_graph=True, allow_unused=True)
                gdf = torch.cat([t.reshape(-1) for t in gd if t is not None])
                gcf = torch.cat([t.reshape(-1) for t in gc if t is not None])
                comp = (float(loss_d), float(loss_c), float(gdf.norm()), float(gcf.norm()),
                        float(torch.dot(gdf, gcf) /
                              (gdf.norm() * gcf.norm()).clamp_min(1e-12)))
                del gd, gc, gdf, gcf
            hcomp = None
            if half_loss is not None:
                hv = []
                for hl in half_loss:  # d_loc, c_loc, d_glob, c_glob
                    if torch.is_tensor(hl) and hl.requires_grad:
                        g = torch.autograd.grad(hl, params, retain_graph=True,
                                                allow_unused=True)
                        hv.append(torch.cat([t.reshape(-1) for t in g if t is not None]))
                    else:
                        hv.append(None)
                if all(v is not None for v in hv):
                    hcomp = ([float(v.norm()) for v in hv],
                             float(torch.dot(hv[0], hv[1]) /
                                   (hv[0].norm() * hv[1].norm()).clamp_min(1e-12)),
                             float(torch.dot(hv[2], hv[3]) /
                                   (hv[2].norm() * hv[3].norm()).clamp_min(1e-12)))
                del hv
            rel_comp = None
            if rel_rot is not None and rel_rot.requires_grad:
                gr = torch.autograd.grad(rel_rot, params, retain_graph=True, allow_unused=True)
                gt_ = torch.autograd.grad(rel_tdir, params, retain_graph=True, allow_unused=True)
                grf = torch.cat([t.reshape(-1) for t in gr if t is not None])
                gtf = torch.cat([t.reshape(-1) for t in gt_ if t is not None])
                rel_comp = (float(rel_rot), float(rel_tdir),
                            float(grf.norm()), float(gtf.norm()))
                del gr, gt_, grf, gtf
            (loss / args.accum).backward()
            visit += 1
        max_norm = P.CLIP if args.clip is None else (
            float("inf") if args.clip == 0 else args.clip)
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        if comp is not None:
            l_d, l_c, g_d, g_c, gdir = comp
            share_c = g_c / (g_d + g_c) if (g_d + g_c) > 0 else float("nan")
            grad_rows.append((upd + 1, pi_last, l_d, l_c, g_d, g_c, share_c, gdir,
                              float(gn), lr_now))
            htxt = ""
            if hcomp is not None:
                (gl_m, gl_c, gg_m, gg_c), dirl, dirg = hcomp
                sl = gl_c / (gl_m + gl_c) if (gl_m + gl_c) > 0 else float("nan")
                sg = gg_c / (gg_m + gg_c) if (gg_m + gg_c) > 0 else float("nan")
                half_rows.append((upd + 1, pi_last, gl_m, gl_c, sl, dirl,
                                  gg_m, gg_c, sg, dirg))
                htxt = (f" | loc: gm={gl_m:.4f} gc={gl_c:.4f} cs={sl:.3f} dir={dirl:.3f}"
                        f" | glob: gm={gg_m:.4f} gc={gg_c:.4f} cs={sg:.3f} dir={dirg:.3f}")
            rtxt = ""
            if rel_comp is not None:
                lr_, lt_, gr_, gt2_ = rel_comp
                rel_rows.append((upd + 1, pi_last, lr_, lt_, gr_, gt2_))
                rtxt = (f" | rot: L={lr_:.4f} g={gr_:.4f}"
                        f" | tdir: L={lt_:.4f} g={gt2_:.4f}")
            print(f"[{args.scene}] update {upd+1}/{args.updates} pair={pi_last} "
                  f"L_mse={l_d:.4f} L_cos={l_c:.4f} g_mse={g_d:.4f} g_cos={g_c:.4f} "
                  f"cos_share={share_c:.3f} dir={gdir:.3f} gn={float(gn):.3f} "
                  f"lr={lr_now:.2e} "
                  f"peak={torch.cuda.max_memory_allocated()/2**20:.0f}MiB{htxt}{rtxt}",
                  flush=True)
        else:
            print(f"[{args.scene}] update {upd+1}/{args.updates} "
                  f"last-pair loss={float(loss):.4f} gn={float(gn):.3f} "
                  f"lr={lr_now:.2e} "
                  f"peak={torch.cuda.max_memory_allocated()/2**20:.0f}MiB", flush=True)
    print(f"[{args.scene}] {visit} pair visits, {time.time()-t0:.0f}s")

    if grad_rows:
        csv_path = os.path.join(out, "grad_components.csv")
        with open(csv_path, "w") as f:
            f.write("update,pair,l_mse,l_cos,g_mse,g_cos,cos_share,dir,gn,lr\n")
            for u, p_, l_d, l_c, g_d, g_c, sh, dr, g_, lr_ in grad_rows:
                f.write(f"{u},{p_},{l_d:.6f},{l_c:.6f},{g_d:.6f},{g_c:.6f},"
                        f"{sh:.6f},{dr:.6f},{g_:.6f},{lr_:.3e}\n")
        print(f"[{args.scene}] per-component grads -> {csv_path}")
    if half_rows:
        csv2 = os.path.join(out, "grad_components_halves.csv")
        with open(csv2, "w") as f:
            f.write("update,pair,g_mse_loc,g_cos_loc,cos_share_loc,dir_loc,"
                    "g_mse_glob,g_cos_glob,cos_share_glob,dir_glob\n")
            for u, p_, glm, glc, sl, dl, ggm, ggc, sg, dg in half_rows:
                f.write(f"{u},{p_},{glm:.6f},{glc:.6f},{sl:.6f},{dl:.6f},"
                        f"{ggm:.6f},{ggc:.6f},{sg:.6f},{dg:.6f}\n")
        print(f"[{args.scene}] per-half grads -> {csv2}")
    if rel_rows:
        csv3 = os.path.join(out, "grad_rel.csv")
        with open(csv3, "w") as f:
            f.write("update,pair,l_rot,l_tdir,g_rot,g_tdir\n")
            for u, p_, lr_, lt_, gr_, gt_ in rel_rows:
                f.write(f"{u},{p_},{lr_:.6f},{lt_:.6f},{gr_:.6f},{gt_:.6f}\n")
        print(f"[{args.scene}] rel-pose grads -> {csv3}")

    ckpt = os.path.join(out, "ckpts", args.scene, "c2m_final_lora.pt")
    student.save_lora_weights(ckpt)

    # standard cam_dec evaluation with the trained adapter
    P.reset_lora_(student); student.load_lora_weights(ckpt)
    student.to(device).eval()
    ev = evaluate_scene(student, sd, proto["eval_frames"], scene=args.scene,
                        dataset_obj=ds_obj, export_dir=os.path.join(out, "recon", args.scene))
    print(f"[{args.scene}] {args.loss_form}/{args.half_mode}"
          + ("/enc" if args.space == "enc" else "")
          + ("/noclip" if args.clip == 0 else "")
          + f" u{args.updates}k{args.accum} (cam_dec): "
          f"AUC={ev['auc03']:.4f} F1={ev['recon_fscore']:.4f} "
          f"abs_rel={ev.get('abs_rel', float('nan')):.4f}")


if __name__ == "__main__":
    main()
