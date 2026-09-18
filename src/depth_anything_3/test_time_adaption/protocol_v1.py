"""Free-Geometry TTA protocol v1 (2026-09-13) ported to DA3-giant-1.1.

Port of the VGGT protocol in docs/TTA_PROTOCOL_v1_2026-09-13.md whose reference
implementation lives in diagnostics/free_geometry/{train_arms,modeling,common,
build_final_manifest}.py. This module is self-contained (zero GT at selection and
training time) and reuses the existing scaffold
(depth_anything_3.test_time_adaption.models.StudentModel/TeacherModel) only for
LoRA injection and model loading.

DA3-giant-1.1 interface facts this port relies on (audited 2026-09-13):
- Backbone: DinoV2 vitg, 40 blocks, embed_dim 1536, alt_start=13 (cross-view
  "global" attention at odd layers >= 13), cat_token=True.
- get_intermediate_layers(n=<list>) returns one (features, camera_token) tuple
  per tapped layer; features [B,S,P,3072] = cat([raw local track,
  final-LN(global track)]); token 0 (cls/camera) and register tokens are already
  stripped (patch_start_idx equivalent = 0; num_register_tokens = 0).
  camera_token [B,S,3072] is the RAW position-0 token (pre final norm).
- DualDPT head has a SHARED token LayerNorm: head.norm = LayerNorm(3072),
  applied identically to every tapped layer -> the post-LN space exists and is
  used for maskdistill, mirroring VGGT's depth_head.norm.
- Camera decoder: CameraDec(3072 -> 9) on the LAST tapped layer's camera token;
  pose_enc = [T(3), quat xyzw(4), fov_h, fov_w]; pose_encoding_to_extri_intri
  decodes to c2w; the API stores extrinsics = affine_inverse(c2w) = w2c.
- Head taps (out_layers) = [19, 27, 33, 39]; the DualDPT head must receive
  exactly those four (order matters), so the tapped union list is
  [6, 19, 26, 27, 33, 39] and the head gets positions [1, 3, 4, 5].
- ref_view_strategy="first" makes the reference-view reordering the identity
  (deterministic); collection restores the original view order anyway.
"""

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse

# ---------------------------------------------------------------------------
# Protocol constants (identical to the VGGT side)
# ---------------------------------------------------------------------------
TAU_THRESHOLD = 0.55
OVERLAP_LO, OVERLAP_HI = 0.1, 0.5
N_TRAIN_PAIRS, N_PROBE_PAIRS = 10, 2
STUDENT_SLOTS = [0, 2, 4, 6]  # shared frames sit at these teacher slots
PROCESS_RES = 504
PATCH_SIZE = 14

# DA3-giant: 40 blocks. Feature-distill taps MUST equal the DualDPT head's own
# out_layers so every layer the head consumes is supervised (v2 fix: was
# [6, 19, 26, 39], leaving 27/33 unsupervised and supervising unused 6/26).
DA3_NUM_LAYERS = 40
TAP_LAYERS = [19, 27, 33, 39]
HEAD_OUT_LAYERS = [19, 27, 33, 39]  # fixed by the DA3-giant config (DualDPT taps)
TAP_UNION = sorted(set(TAP_LAYERS) | set(HEAD_OUT_LAYERS))  # teacher taps: [19, 27, 33, 39]

# Training hyperparameters (protocol section 3)
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 32, 32.0, 0.0
LR, WD, CLIP = 3e-5, 1e-5, 1.0
EPOCHS = 10  # 10 pairs x 10 epochs = 100 steps
WARMUP_RATIO = 0.15


def stable_seed(*parts) -> int:
    key = "::".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Frame selection (GT-free; ported from build_final_manifest.py)
# ---------------------------------------------------------------------------
def sift_one(path: str, max_side: int = 320):
    sift = cv2.SIFT_create(nfeatures=400)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    sc = max_side / max(h, w)
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)))
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def match_frac(feats_i, feats_j) -> float:
    kpi, di = feats_i
    kpj, dj = feats_j
    if di is None or dj is None or len(kpi) < 10 or len(kpj) < 10:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(di, dj, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    return len(good) / max(1, min(len(kpi), len(kpj)))


class SiftCache:
    def __init__(self, image_files: Sequence[str]):
        self.image_files = list(image_files)
        self.cache: Dict[int, object] = {}

    def feats(self, i: int):
        if i not in self.cache:
            self.cache[i] = sift_one(self.image_files[i])
        return self.cache[i]

    def frac(self, i: int, j: int) -> float:
        return match_frac(self.feats(i), self.feats(j))


def compute_tau(image_files: Sequence[str]) -> float:
    """Median adjacent-frame SIFT match fraction over <=40 equidistant pairs."""
    N = len(image_files)
    if N < 2:
        return 0.0
    sc = SiftCache(image_files)
    n_pairs = min(40, N - 1)
    starts = np.linspace(0, N - 2, n_pairs).astype(int)
    fr = [sc.frac(int(i), int(i) + 1) for i in starts]
    return float(np.median(fr))


def assemble_teacher_list(shared: List[int], extras: List[int]) -> List[int]:
    """Shared (ascending) at even slots [0,2,4,...]; extras (ascending) fill the rest."""
    teacher = [None] * (len(shared) + len(extras))
    slots = list(range(0, 2 * len(shared), 2))
    for slot, s in zip(slots, shared):
        teacher[slot] = s
    rest = [i for i in range(len(teacher)) if i not in slots]
    for slot, e in zip(rest, extras):
        teacher[slot] = e
    return teacher


def build_pair(N: int, teacher_N: int, dense: bool, sc: Optional[SiftCache], rng: random.Random,
               n_shared: int = 4, selfevo: bool = False, selfevo_lmin: int = 2,
               sparse_overlap: bool = False):
    """One (teacher_frames, student_frames) pair following the tau dispatch.
    n_shared=4 -> 16:4 protocol (student slots [0,2,4,6]); n_shared=8 -> 16:8.
    selfevo=True (SelfEvo-faithful): student = teacher window's FIRST + LAST frame
    plus L-2 random middle frames, L ~ U{selfevo_lmin, min(6, teacher_N//2)} per pair
    (endpoint-anchored, randomized asymmetry level).
    sparse_overlap=True: on the tau<=0.55 (random) branch, resample the window until
    every teacher frame's mean SIFT match fraction vs the shared frames >= OVERLAP_LO
    (soft fallback: best of 40 tries). Targets wide-baseline scenes where a purely
    random window mixes mutually-invisible frames and degrades the teacher."""
    if dense:
        M = min(N, teacher_N * 3 // 2)
        step = N / M
        off = rng.uniform(0, step)
        sup = sorted(set(int(off + i * step) for i in range(M)))[:M]
        if selfevo:
            l_max = max(selfevo_lmin, min(6, teacher_N // 2, len(sup) - 2))
            L = rng.randint(selfevo_lmin, l_max)
            L = min(L, len(sup))
            mids = rng.sample(sup[1:-1], k=max(0, L - 2))
            shared = sorted([sup[0]] + mids + [sup[-1]])
        else:
            stride = max(1, M // n_shared)
            shared = sup[::stride][:n_shared]
        rest = [f for f in sup if f not in set(shared)]
        in_range, out_range = [], []
        for f in rest:
            c = float(np.mean([sc.frac(f, s) for s in shared]))
            (in_range if OVERLAP_LO <= c <= OVERLAP_HI else out_range).append(f)
        extras = sorted((in_range + out_range)[: teacher_N - len(shared)])
    else:
        def draw():
            t = sorted(rng.sample(range(N), teacher_N))
            if selfevo:
                l_max = max(selfevo_lmin, min(6, teacher_N // 2, len(t) - 2))
                L = rng.randint(selfevo_lmin, l_max)
                L = min(L, len(t))
                mids = rng.sample(t[1:-1], k=max(0, L - 2))
                sh = sorted([t[0]] + mids + [t[-1]])
            else:
                stride = max(1, teacher_N // n_shared)
                sh = t[::stride][:n_shared]
            return t, sh
        if not (sparse_overlap and sc is not None):
            t, shared = draw()
        else:
            # covisibility-constrained resampling: every teacher frame must
            # (on average) share SIFT content with the student's shared frames
            best, best_score = None, -1.0
            for _ in range(40):
                t_c, sh_c = draw()
                fr = [float(np.mean([sc.frac(f, s) for s in sh_c])) for f in t_c]
                score = min(fr)
                if score >= OVERLAP_LO:
                    best = (t_c, sh_c)
                    break
                if score > best_score:
                    best, best_score = (t_c, sh_c), score
            t, shared = best
        extras = [f for f in t if f not in set(shared)]
    shared = sorted(shared)
    teacher = assemble_teacher_list(shared, extras)
    return teacher, shared


def build_scene_protocol(
    image_files: Sequence[str],
    scene: str,
    dataset: str = "scannetpp",
    n_train: int = N_TRAIN_PAIRS,
    n_probe: int = N_PROBE_PAIRS,
    n_shared: int = 4,
    teacher_N: Optional[int] = None,
    ratio_mix: Optional[List[Tuple[int, int]]] = None,
    selfevo: bool = False,
    combo: bool = False,
    combo_se_frac: float = 0.5,
    sparse_overlap: bool = False,
) -> Dict:
    """Full GT-free per-scene protocol: tau, teacher_N, 10+2 pairs, eval frames.
    teacher_N override enables ratio variants: 16:4 (default), 16:8, 8:4, 8:2, 32:8.
    ratio_mix=[(t1,s1),(t2,s2),...] samples a (teacher_N, n_shared) ratio per pair
    (SelfEvo-style mixed context asymmetry within ONE training run).
    selfevo=True: endpoint-anchored student with per-pair random L (SelfEvo-faithful).
    combo=True: each pair is seeded-either fixed-slot (camera-oriented) or
    endpoint-anchored with L>=4 (fusion-oriented), drawn with probability
    combo_se_frac (0.5 = 1:1, 0.25 = 3:1 fixed:SE).
    sparse_overlap=True: covisibility-constrained teacher windows on the random
    branch (see build_pair)."""
    image_files = list(image_files)
    N = len(image_files)
    if N < 8:
        raise ValueError(f"{scene}: N={N} < 8, scene dropped by protocol")
    if teacher_N is None:
        teacher_N = 16 if N >= 16 else 8
    teacher_N = min(teacher_N, N)
    if teacher_N <= n_shared:
        raise ValueError(f"{scene}: teacher_N={teacher_N} <= n_shared={n_shared}")
    tau = compute_tau(image_files)
    dense = tau > TAU_THRESHOLD
    sc = SiftCache(image_files) if (dense or sparse_overlap) else None

    def sample(tag: str, n: int, forb_t: set, forb_s: set):
        r = random.Random(stable_seed("final", tag, dataset, scene))
        pairs, tries = [], 0
        while len(pairs) < n and tries < 10000:
            tries += 1
            if ratio_mix:
                tn, ns = ratio_mix[r.randrange(len(ratio_mix))]
                tn = min(tn, N)
                if tn <= ns:
                    continue
            else:
                tn, ns = teacher_N, n_shared
            ns = min(ns, tn // 2)  # shared slots [0,2,...] must fit the teacher list
            se = selfevo or (combo and r.random() < combo_se_frac)
            teacher, shared = build_pair(N, tn, dense, sc, r, n_shared=ns,
                                         selfevo=se, selfevo_lmin=(4 if combo else 2),
                                         sparse_overlap=sparse_overlap)
            key_t, key_s = tuple(sorted(teacher)), tuple(shared)
            if key_t in forb_t or key_s in forb_s:
                continue
            forb_t.add(key_t)
            forb_s.add(key_s)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared),
                          "teacher_N": tn, "n_shared": ns,
                          "kind": "se" if se else "fixed"})
        while len(pairs) < n:  # tiny scene: pair space exhausted
            tn, ns = (ratio_mix[r.randrange(len(ratio_mix))] if ratio_mix
                      else (teacher_N, n_shared))
            tn = min(tn, N)
            ns = min(ns, tn // 2)
            se = selfevo or (combo and r.random() < combo_se_frac)
            teacher, shared = build_pair(N, tn, dense, sc, r, n_shared=ns,
                                         selfevo=se, selfevo_lmin=(4 if combo else 2),
                                         sparse_overlap=sparse_overlap)
            pairs.append({"teacher_frames": teacher, "student_frames": list(shared),
                          "teacher_N": tn, "n_shared": ns,
                          "kind": "se" if se else "fixed"})
        return pairs

    train_pairs = sample("train", n_train, set(), set())
    probe_pairs = sample(
        "probe",
        n_probe,
        {tuple(sorted(p["teacher_frames"])) for p in train_pairs},
        {tuple(p["student_frames"]) for p in train_pairs},
    )

    if N >= 100:
        r = random.Random(42)
        idx = list(range(N))
        r.shuffle(idx)
        eval_frames = sorted(idx[:100])
    else:
        eval_frames = list(range(N))

    return {
        "scene": scene,
        "dataset": dataset,
        "N": N,
        "teacher_N": teacher_N,
        "tau": tau,
        "strategy": ("dense_equidistant_sift" if dense
                     else ("random_overlap" if sparse_overlap else "random")),
        "train_pairs": train_pairs,
        "probe_pairs": probe_pairs,
        "eval_frames": eval_frames,
        "n_shared": n_shared,
        "student_slots": list(range(0, 2 * n_shared, 2)),
    }


# ---------------------------------------------------------------------------
# Image loading (identical to the DA3 inference path: InputProcessor,
# upper_bound_resize @ 504, round to 14, ImageNet normalize)
# ---------------------------------------------------------------------------
def load_images_da3(image_files: Sequence[str], process_res: int = PROCESS_RES) -> torch.Tensor:
    """[N,3,H,W] fp32 CPU tensor, ImageNet-normalized (as the model expects)."""
    from depth_anything_3.utils.io.input_processor import InputProcessor

    proc = InputProcessor()
    imgs, _, _ = proc(
        list(image_files), None, None, process_res, "upper_bound_resize", sequential=True
    )
    return imgs  # [N,3,H,W]


# ---------------------------------------------------------------------------
# Masking + loss forms (ported verbatim from train_arms.py)
# ---------------------------------------------------------------------------
def mask_image_blocks(images: torch.Tensor, ratio: float, patch_hw: Tuple[int, int], gen):
    """Zero random 14x14 blocks of the (normalized) input. Returns
    (masked_images, patch_mask [1,S,P], 1=masked)."""
    B, S, C, H, W = images.shape
    ph, pw = patch_hw
    block = torch.rand(S, ph, pw, generator=gen, device=images.device) < ratio
    m = block.repeat_interleave(H // ph, dim=1).repeat_interleave(W // pw, dim=2)
    out = images * (~m)[:, None].float()
    return out, block.reshape(1, S, ph * pw).float()


def teacher_patch_conf(conf4: torch.Tensor, patch_hw: Tuple[int, int]) -> torch.Tensor:
    """Teacher depth_conf [1,4,H,W] -> per-patch weight [1,4,P], mean 1."""
    conf = conf4.float()
    if conf.dim() == 5:
        conf = conf.squeeze(2)
    B, S, H, W = conf.shape
    ph, pw = patch_hw
    confp = F.avg_pool2d(conf.reshape(B * S, 1, H, W), kernel_size=(H // ph, W // pw))
    confp = confp.reshape(B, S, ph * pw)
    w = confp / confp.mean().clamp_min(1e-8)
    return w.detach()


def _huber_cos(hs: torch.Tensor, ht: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Deployed patch-loss form: Huber(beta=1)*1 + 2*(1-cos), per-patch weighted.
    hs/ht: [B,S,P,C]; w: [B,S,P] pre-normalized weights."""
    huber_t = F.smooth_l1_loss(hs, ht, beta=1.0, reduction="none").mean(dim=-1)
    cos_t = F.cosine_similarity(hs, ht, dim=-1)
    huber = (huber_t * w).mean()
    cos = (cos_t * w).mean()
    return huber + 2.0 * (1.0 - cos)


def loss_maskdistill(
    head_norm: torch.nn.Module,
    teacher_cache: Dict,
    tap_feats_s: Dict[int, torch.Tensor],
    patch_mask: torch.Tensor,
    q_feat: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """C2M maskdistill for DA3: post-LN (DualDPT shared norm) Huber+2cos on the
    4 tapped layers, teacher-conf weighted, masked positions only.

    teacher_cache["feats"][layer]: [1,4,P,3072] raw concat tokens of the shared
    views (bf16 cache, upcast here); tap_feats_s[layer]: [1,4,P,3072] student.

    q_feat (optional, AB reliability): frozen per-patch weights [4,P] in [0,1].
    When given, w = conf_mean1 * q_feat * patch_mask with NO renormalization —
    renormalizing would cancel exactly the downweighting q encodes. The legacy
    path (q_feat=None) renormalizes after masking, unchanged."""
    w = teacher_patch_conf(teacher_cache["conf4"], teacher_cache["patch_hw"])
    if q_feat is not None:
        w = w * q_feat.to(w.device).float()
    else:
        w = w / w.mean().clamp_min(1e-8)
    w = w * patch_mask
    total = 0.0
    for layer in TAP_LAYERS:
        hs = head_norm(tap_feats_s[layer].float())
        ht = head_norm(teacher_cache["feats"][layer].float()).detach()
        total = total + _huber_cos(hs, ht, w)
    loss = total / len(TAP_LAYERS)
    return loss, {"mask_ratio": float(patch_mask.mean()), "maskdistill": float(loss)}


def loss_ctk(cam_s: Dict[int, torch.Tensor],
             cam_t: Dict[int, torch.Tensor], space: str = "raw"
             ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Camera-token KD on the shared views: Huber + 2(1-cos), averaged over
    TAP_LAYERS. cam_s/cam_t: {layer: [1,S,3072]} raw position-0 tokens
    (student grad, teacher detached).

    space="raw": compare the raw tokens directly — the space DA3's deployed
    CameraDec actually reads (da3.py: cam_dec(feats[-1][1]) with no norm in
    front), and the honest counterpart of VGGT's token_norm space.
    space="ln_free": parameter-free LayerNorm on both sides (no learned
    affine) — neutral channel scaling if raw magnitudes are outlier-heavy.
    (The earlier "head_norm" space was dropped: DualDPT's shared norm is
    calibrated for patch tokens and its affine distorts per-channel weights
    when applied to camera tokens — B1 confound.)"""
    total = 0.0
    for layer in TAP_LAYERS:
        hs = cam_s[layer].float()
        ht = cam_t[layer].float().detach()
        if space == "ln_free":
            hs = F.layer_norm(hs, (hs.shape[-1],))
            ht = F.layer_norm(ht, (ht.shape[-1],))
        hs = hs.unsqueeze(2)  # [1,S,1,C]
        ht = ht.unsqueeze(2)
        w = torch.ones(hs.shape[:3], device=hs.device, dtype=hs.dtype)
        total = total + _huber_cos(hs, ht, w)
    loss = total / len(TAP_LAYERS)
    return loss, {"ctk": float(loss)}


def loss_pose_rel(ext_s_w2c: torch.Tensor, ext_t_w2c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Relative-pose loss, CORRECTED w2c construction (2026-09-18):
    T_{i<-j} = E_i @ inv(E_j) (NOT inv(E_i) @ E_j).
    R_rel = R_i @ R_j^T,  t_rel = t_i - R_i @ R_j^T @ t_j.
    Per view-pair rotation chordal + translation-direction 1-cos, scale-free.
    ext_*: [S,4,4] or [1,S,4,4] w2c extrinsics (student grad, teacher detached)."""
    if ext_s_w2c.dim() == 4:
        ext_s_w2c = ext_s_w2c[0]
    if ext_t_w2c.dim() == 4:
        ext_t_w2c = ext_t_w2c[0]
    R_s, t_s = ext_s_w2c[..., :3, :3], ext_s_w2c[..., :3, 3]
    R_t, t_t = ext_t_w2c[..., :3, :3], ext_t_w2c[..., :3, 3]
    S = R_s.shape[0]
    rot_loss, tdir_loss, npairs = 0.0, 0.0, 0
    for i in range(S):
        for j in range(i + 1, S):
            # Correct: T_{i<-j} = E_i @ inv(E_j)
            Rr_s = R_s[i] @ R_s[j].transpose(-1, -2)
            Rr_t = R_t[i] @ R_t[j].transpose(-1, -2)
            tr_s = (t_s[i] - Rr_s @ t_s[j])[..., None] if t_s.dim() == 2 else t_s[i] - Rr_s @ t_s[j]
            tr_t = (t_t[i] - Rr_t @ t_t[j])[..., None] if t_t.dim() == 2 else t_t[i] - Rr_t @ t_t[j]
            rot_loss = rot_loss + ((Rr_s - Rr_t) ** 2).sum(dim=(-2, -1)).mean()
            tn_s = F.normalize(tr_s.squeeze(-1), dim=-1, eps=1e-8)
            tn_t = F.normalize(tr_t.squeeze(-1), dim=-1, eps=1e-8)
            tdir_loss = tdir_loss + (1.0 - (tn_s * tn_t).sum(-1)).mean()
            npairs += 1
    rot_loss = rot_loss / npairs
    tdir_loss = tdir_loss / npairs
    return rot_loss + tdir_loss, {"rel_rot": float(rot_loss), "rel_tdir": float(tdir_loss)}


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------
def _unwrap_pretrained(backbone) -> torch.nn.Module:
    """DinoV2.pretrained, unwrapping PEFT if present (LoRA lives inside)."""
    pre = backbone.pretrained
    if hasattr(pre, "base_model"):  # PeftModel -> LoraModel -> DinoVisionTransformer
        base = pre.base_model
        base = base.model if hasattr(base, "model") else base
        return base
    return pre


def backbone_tapped_forward(da3_net, images: torch.Tensor, layers: Sequence[int],
                            ref_view_strategy: str = "first"):
    """One backbone forward tapping `layers` (ascending).

    Returns (feats, H, W): feats is a list of (features [B,S,P,3072],
    camera_token [B,S,3072]) in `layers` order. Works for both the frozen
    teacher and the PEFT-wrapped student (LoRA fires inside the unwrapped base).
    """
    backbone = da3_net.model.backbone
    vit = _unwrap_pretrained(backbone)
    feats, _aux = vit.get_intermediate_layers(
        images,
        list(layers),
        export_feat_layers=[],
        cam_token=None,
        ref_view_strategy=ref_view_strategy,
    )
    H, W = images.shape[-2], images.shape[-1]
    return list(feats), H, W


def split_tap_feats(feats, layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    """{layer: features [B,S,P,3072]} for the requested tap layers."""
    return {layer: feats[i][0] for i, layer in enumerate(layers)}


def head_feats(feats, layers: Sequence[int]):
    """The four (features, camera_token) tuples the DualDPT head expects.
    `layers` must be a superset of HEAD_OUT_LAYERS (ascending)."""
    pos = {l: i for i, l in enumerate(layers)}
    return [feats[pos[l]] for l in HEAD_OUT_LAYERS]


def decode_pose_w2c(cam_dec: torch.nn.Module, cam_token_last: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """cam_token_last: [B,S,3072] raw position-0 token of the last tapped layer.
    Returns w2c extrinsics [B,S,3,4] (grad flows through the frozen cam_dec)."""
    pose_enc = cam_dec(cam_token_last)  # [B,S,9] = T + quat(xyzw) + fov
    c2w, _ixt = pose_encoding_to_extri_intri(pose_enc.float(), (H, W))
    return affine_inverse(c2w)  # [B,S,4,4] w2c


def create_teacher(model_name: str, device: str = "cpu") -> DepthAnything3:
    da3 = DepthAnything3.from_pretrained(model_name)
    da3.eval()
    for p in da3.parameters():
        p.requires_grad = False
    return da3.to(device)


def create_student(
    model_name: str,
    device: str = "cpu",
    lora_rank: int = LORA_RANK,
    lora_alpha: float = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
):
    """StudentModel with protocol LoRA: r=32/a=32/dropout=0 on the multi-view
    blocks 13..39 ONLY (attn qkv/proj + SwiGLU w12/w3); heads frozen, camera
    token trainable. Layers 0..12 — the per-view "DINO" local-attention blocks
    before alt_start (camera-token injection / first cross-view attention) —
    are STRICTLY FROZEN: no LoRA there, by explicit decision (2026-09-18)."""
    from depth_anything_3.test_time_adaption.models import StudentModel

    student = StudentModel(
        model_name=model_name,
        output_layers=list(HEAD_OUT_LAYERS),
        embed_dim=1536,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_camera_token=True,  # v2: camera token trainable (was frozen in v1);
                                  # cam_dec + heads stay frozen
        lora_layers=list(range(13, 40)),  # multi-view scope ONLY (>= alt_start)
        ref_view_strategy="first",
        patch_swiglu_mlp_for_lora=True,
    )
    base = _unwrap_pretrained(student.da3.model.backbone)
    if hasattr(base, "camera_token"):
        student._camera_token_init = base.camera_token.detach().clone()
    return student.to(device)


def reset_lora_(student) -> None:
    """In-place PEFT-faithful LoRA re-init: A ~ kaiming_uniform(sqrt5), B = 0
    (student == frozen baseline right after the reset)."""
    import torch.nn as nn

    base = _unwrap_pretrained(student.da3.model.backbone)
    count = 0
    for m in base.modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
            for a in m.lora_A.values():
                nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
                count += 1
            for b in m.lora_B.values():
                nn.init.zeros_(b.weight)
    assert count > 0, "no LoRA modules found"
    # v2: camera token is trainable — restore its pre-scene init so every scene
    # starts from the base model (per-scene protocol, same as the LoRA reset).
    if hasattr(base, "camera_token") and getattr(student, "_camera_token_init", None) is not None:
        base.camera_token.data.copy_(student._camera_token_init.to(base.camera_token.device))


# ---------------------------------------------------------------------------
# Teacher cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_teacher_pair(
    teacher: DepthAnything3,
    images_t: torch.Tensor,
    shared_slots: Sequence[int],
    patch_hw: Tuple[int, int],
    ref_view_strategy: str = "first",
) -> Dict:
    """One teacher_N-view frozen forward. Caches (CPU, moved to GPU in the loss):
    - feats: {layer: [1,4,P,3072] bf16} raw concat tokens at TAP_LAYERS, shared
      views only (the LN is applied inside the loss, as on the VGGT side)
    - conf4: [1,4,H,W] fp32 teacher depth_conf on the shared views
    - ext4: [4,3,4] fp32 teacher w2c extrinsics on the shared views
    - cam: {layer: [1,4,3072] bf16} raw position-0 camera tokens at TAP_LAYERS,
      shared views only (for the ctk camera-token KD term)
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats, H, W = backbone_tapped_forward(teacher, images_t, TAP_UNION, ref_view_strategy)
    tap = split_tap_feats(feats, TAP_LAYERS)
    cache_feats = {
        layer: t[:, list(shared_slots)].detach().to(torch.bfloat16).cpu()
        for layer, t in tap.items()
    }
    cache_cam = {
        layer: feats[i][1][:, list(shared_slots)].detach().to(torch.bfloat16).cpu()
        for i, layer in enumerate(TAP_UNION)
    }

    hf = [(f.float(), c.float()) for f, c in head_feats(feats, TAP_UNION)]
    with torch.autocast(device_type="cuda", enabled=False):
        preds = teacher.model.forward_head_only(hf, H=H, W=W, process_camera=True, process_sky=False)
    conf = preds["depth_conf"][:, list(shared_slots)].float().cpu()  # [1,4,H,W]
    ext = preds["extrinsics"][0, list(shared_slots)].float().cpu()  # [4,3,4] w2c
    depth4 = preds["depth"][:, list(shared_slots)].float().cpu()  # [1,4,H,W] (couple term)
    return {
        "feats": cache_feats,
        "cam": cache_cam,
        "conf4": conf.detach(),
        "ext4": ext.detach(),
        "depth4": depth4.detach(),
        "patch_hw": patch_hw,
    }


# ---------------------------------------------------------------------------
# Student forward (grad) + C2M loss
# ---------------------------------------------------------------------------
def student_forward_c2m(student, images4: torch.Tensor, ref_view_strategy: str = "first",
                        with_depth: bool = False, with_cam: bool = False):
    """Student 4-view forward under bf16 autocast, tapping TAP_LAYERS
    (== HEAD_OUT_LAYERS since the v2 tap fix). Returns (tap_feats, ext_w2c,
    depth_s[, tap_cam]). Only the frozen CameraDec runs by default;
    with_depth=True also runs the frozen DualDPT head (needed by the couple
    term's student depth stat; head weights frozen, grads flow through to the
    tapped features). with_cam=True also returns {layer: camera_token
    [1,S,3072]} (grad) for the ctk term."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats, H, W = backbone_tapped_forward(student.da3, images4, TAP_LAYERS, ref_view_strategy)
    tap = split_tap_feats(feats, TAP_LAYERS)
    cam_token_last = feats[-1][1]  # raw [1,4,3072] at layer 39 (last tap)
    with torch.autocast(device_type="cuda", enabled=False):
        ext_w2c = decode_pose_w2c(student.da3.model.cam_dec, cam_token_last.float(), H, W)
    depth_s = None
    if with_depth:
        hf = [(f.float(), c.float()) for f, c in head_feats(feats, TAP_LAYERS)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = student.da3.model.forward_head_only(hf, H=H, W=W,
                                                        process_camera=False, process_sky=False)
        depth_s = preds["depth"]
    if with_cam:
        tap_cam = {layer: feats[i][1] for i, layer in enumerate(TAP_LAYERS)}
        return tap, ext_w2c, depth_s, tap_cam
    return tap, ext_w2c, depth_s


def loss_rkd_shared_pose_huber_w2c(ext_s: torch.Tensor, ext_t: torch.Tensor,
                                   delta: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Anchor-free RKD on the 4 shared frames' camera centers (gauge-free), the
    DA3 port of VGGT loss_rkd_shared_pose_huber. Mean-normalized 6 pairwise
    distances + 12 triangle angles, per-residual Huber(delta). ext_*: [S,3,4]
    or [1,S,3,4] w2c (student grad, teacher detached)."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(ext):
            if ext.dim() == 4:
                ext = ext[0]
            R, t = ext[..., :3, :3].float(), ext[..., :3, 3].float()
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)

        cs, ct = centers(ext_s), centers(ext_t).detach()
        S = cs.shape[0]
        if S < 3:
            # no triangle/shape structure with <3 points; rkd is identically
            # zero here (single mean-normalized distance), so skip cleanly
            zero = (cs - ct).sum() * 0.0
            return zero, {"rkd_sh_d": 0.0, "rkd_sh_a": 0.0}
        pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
        ds = torch.stack([(cs[i] - cs[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dt = torch.stack([(ct[i] - ct[j]).norm() for i, j in pairs]).clamp_min(1e-6)
        dn_s = ds / ds.mean().clamp_min(1e-8)
        dn_t = dt / dt.mean().clamp_min(1e-8)
        dloss = F.huber_loss(dn_s, dn_t, delta=delta)
        if S < 4:
            # S=3 (selfevo sampling can yield 3-frame students): distance pairs
            # only — triangle angles need >=4 points (3 others per vertex)
            return dloss, {"rkd_sh_d": float(dloss), "rkd_sh_a": 0.0}
        aterms = []
        for k in range(S):
            rest = [x for x in range(S) if x != k]
            for a, b in [(rest[0], rest[1]), (rest[0], rest[2]), (rest[1], rest[2])]:
                vs1 = F.normalize(cs[a] - cs[k], dim=-1, eps=1e-8)
                vs2 = F.normalize(cs[b] - cs[k], dim=-1, eps=1e-8)
                vt1 = F.normalize(ct[a] - ct[k], dim=-1, eps=1e-8)
                vt2 = F.normalize(ct[b] - ct[k], dim=-1, eps=1e-8)
                aterms.append(F.huber_loss((vs1 * vs2).sum(), (vt1 * vt2).sum(), delta=delta))
        aloss = torch.stack(aterms).mean()
        loss = dloss + aloss
    return loss, {"rkd_sh_d": float(dloss), "rkd_sh_a": float(aloss)}


def loss_couple_w2c(ext_s: torch.Tensor, depth_s: torch.Tensor,
                    ext_t: torch.Tensor, depth_t: torch.Tensor,
                    conf_t: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cross-head gauge coupling scalar (DA3 port of VGGT loss_couple):
    log(RMS_centers / mean_depth) aligned student vs teacher (teacher side
    detached, conf-gated at the 5% quantile). One scalar per forward."""
    with torch.autocast(device_type="cuda", enabled=False):
        def centers(ext):
            if ext.dim() == 4:
                ext = ext[0]
            R, t = ext[..., :3, :3].float(), ext[..., :3, 3].float()
            return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)

        def couple_stat(ext, depth, conf=None):
            c = centers(ext)
            spr = (c - c.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()
            d = depth.squeeze(0).squeeze(-1).float() if depth.dim() >= 4 else depth.float()
            m = torch.isfinite(d) & (d > 0)
            if conf is not None:
                cf = conf.squeeze(0).squeeze(-1).float() if conf.dim() >= 4 else conf.float()
                q = torch.quantile(cf[m].flatten(), 0.05)
                m = m & (cf >= q)
            md = d[m].mean().clamp_min(1e-6)
            return torch.log(spr.clamp_min(1e-6)) - torch.log(md)

        cs = couple_stat(ext_s, depth_s)
        ct = couple_stat(ext_t, depth_t, conf_t).detach()
        loss = (cs - ct) ** 2
    return loss.squeeze(), {"couple": float(loss)}


# ---------------------------------------------------------------------------
# Protocol v2 additions (all default-off; v2=None reproduces v1 exactly)
# ---------------------------------------------------------------------------
@dataclass
class V2Config:
    """Protocol-v2 knobs, wired in from scripts/train_da3_protocol.py.

    probe:        GT-free probe evaluation on the protocol's probe_pairs
                  (their teacher caches are kept instead of discarded).
    probe_every:  evaluate every N optimizer updates (plus the step-0 baseline).
    ckpt:         save LoRA at step 0 and at every probe-cadence step to
                  <ckpt_dir>/<scene>/v2/step{N}_lora.pt (hard-fails if missing).
    rel_weight:   >0 appends rel_weight * (rot_edges_huber + tdir_cos_loss) to
                  the arm loss (teacher detached; replaces the old loss_pose_rel
                  path, which remains available via the rkdc1hr arm).
    grad_cap:     separate backward of base vs v2-rel gradients; ||g_R|| capped
                  at 4*median(||g_R|| over the first 10 updates) (requires
                  rel_weight > 0).
    couple_fix:   couple term uses the shared teacher-derived valid mask on
                  BOTH sides (losses.loss_couple_centers semantics), skipping
                  and logging on degenerate spread / empty valid mask.
    run_dir:      probe trace root -> <run_dir>/probe_trace/<scene>.jsonl.
    ckpt_dir:     v2 checkpoint root -> <ckpt_dir>/<scene>/v2/step{N}_lora.pt.
    """

    probe: bool = False
    probe_every: int = 10
    ckpt: bool = False
    rel_weight: float = 0.0
    grad_cap: bool = False
    couple_fix: bool = False
    run_dir: str = "."
    ckpt_dir: str = ""
    ab: bool = False  # dual-teacher A/B contexts + frozen reliability weights
                      # (--v2_ab_manifest; train pairs must carry teacher_frames_B)
    rel_gate_deg: float = 0.0  # scene-level rel gate: disable the v2 rel branch
                               # when the B=0 unmasked probe rot median exceeds
                               # this (0 = gate off; CLI default 30)
    rot_weight: Optional[float] = None   # R/T split; None -> fall back to rel_weight
    tdir_weight: Optional[float] = None  # R/T split; None -> fall back to rel_weight


def _centers_from_ext(ext: torch.Tensor) -> torch.Tensor:
    """Camera centers [S,3] from w2c ext [S,3,4] | [1,S,3,4] | [1,S,4,4]."""
    if ext.dim() == 4:
        ext = ext[0]
    R, t = ext[..., :3, :3].float(), ext[..., :3, 3].float()
    return -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)


def _v2_rel_branch(ext_s: torch.Tensor, ext_t: torch.Tensor,
                   q_rot: Optional[torch.Tensor] = None,
                   q_tdir: Optional[torch.Tensor] = None):
    """v2 robust relative-pose branch on the shared frames' w2c view-graph
    edges (teacher side detached): rot_edges_huber.loss + tdir_cos_loss.loss.
    Nothing here detaches the student side.

    q_rot / q_tdir: optional frozen AB-reliability edge weights [E] in [0,1]
    (NaN where the teacher baseline is ~0). NaN tdir edges are DROPPED from the
    tdir mean (their directions are undefined); rot edges are always defined.

    Returns (rot_loss, tdir_loss, extra) — the two branches SEPARATE, so the
    caller can weight R and T independently (--v2_rot_weight/--v2_tdir_weight).
    """
    from free_geometry.tta_v2 import edges_from_w2c, rot_edges_huber, tdir_cos_loss

    e_t = edges_from_w2c(ext_t)
    e_s = edges_from_w2c(ext_s)
    dev = e_s["R_rel"].device
    edge_w_r = None if q_rot is None else q_rot.to(dev).float()
    rot = rot_edges_huber(e_s["R_rel"], e_t["R_rel"].detach(), edge_w=edge_w_r)
    if q_tdir is not None:
        qT = q_tdir.to(dev).float()
        keep = torch.isfinite(qT)  # near-zero-baseline edges: undefined tdir
        tdir = tdir_cos_loss(e_s["t_rel"][keep], e_t["t_rel"].detach()[keep],
                             e_t["baseline"][keep], edge_w=qT[keep])
        n_nan = int((~keep).sum())
    else:
        tdir = tdir_cos_loss(e_s["t_rel"], e_t["t_rel"].detach(), e_t["baseline"])
        n_nan = 0
    extra = {"v2_rel_rot": float(rot["loss"]), "v2_rel_tdir": float(tdir["loss"]),
             "v2_rel_tdir_n_kept": int(tdir["n_kept"]),
             "v2_rel_tdir_n_skipped": int(tdir["n_skipped"])}
    if n_nan:
        extra["v2_rel_tdir_n_nan"] = n_nan
    return rot["loss"], tdir["loss"], extra


def _couple_fixed(ext_s: torch.Tensor, depth_s: torch.Tensor,
                  ext_t: torch.Tensor, depth_t: torch.Tensor,
                  conf_t: torch.Tensor):
    """Couple term with the losses.loss_couple_centers semantics (2026-09-17
    fix): BOTH sides take mean depth over the SAME teacher-derived valid-pixel
    mask (conf 5% quantile). Degenerate camera spread / empty valid mask ->
    zero loss + skip reason in extra (logged by the caller)."""
    from free_geometry.losses import valid_mask_from_conf
    from free_geometry.tta_v2 import couple_robust

    with torch.autocast(device_type="cuda", enabled=False):
        valid_t = valid_mask_from_conf(conf_t[0].float())  # conf [1,S,H,W] -> [S,H,W]
        cs, ct = _centers_from_ext(ext_s), _centers_from_ext(ext_t).detach()
        ds, dt = depth_s.float(), depth_t.float()
        if ds.dim() == 4:
            ds = ds[0]
        if dt.dim() == 4:
            dt = dt[0]
        cp = couple_robust(cs, ct, ds, dt, valid_t, huber_delta=None)
    if cp["skipped"]:
        zero = (cs - ct).sum() * 0.0
        return zero, {"couple": 0.0, "couple_skipped": cp["reason"]}
    return cp["loss"], {"couple": float(cp["loss"])}


def _save_v2_ckpt(student, ckpt_dir: str, scene: str, step: int) -> str:
    """Save LoRA (+ trainable camera token) to <ckpt_dir>/<scene>/v2/step{N}_lora.pt
    and HARD-FAIL if the artifact is not on disk afterwards — a silently missing
    ckpt once let eval fall back to base and polluted 12 scenes."""
    d = os.path.join(ckpt_dir, scene, "v2")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"step{step}_lora.pt")
    student.save_lora_weights(path)
    assert os.path.isdir(path.replace(".pt", "_peft")), f"v2 ckpt save failed: {path}"
    assert os.path.exists(path), f"v2 ckpt save failed: {path}"
    return path


def _build_probe_evaluator(student, scene: str, protocol: Dict,
                           probe_caches: List[Dict], probe_images: List[torch.Tensor],
                           device: str):
    """ProbeEvaluator contexts from the protocol's probe pairs. The teacher side
    of each context comes from the pair's (previously discarded) teacher cache:
    post-head.norm tapped features, conf patch weights, w2c ext / centers /
    depth / teacher-derived valid mask — all on `device` (the core does no
    device moves). images_meta stays a CPU [4,3,H,W] handle for forward_fn."""
    from free_geometry.losses import valid_mask_from_conf
    from free_geometry.tta_v2 import ProbeEvaluator

    head_norm = student.da3.model.head.norm
    contexts = []
    with torch.no_grad():
        for pi, (cache, imgs4) in enumerate(zip(probe_caches, probe_images)):
            conf = cache["conf4"].to(device)  # [1,4,H,W] fp32
            feats_t = {l: head_norm(cache["feats"][l].to(device).float()).detach()
                       for l in TAP_LAYERS}
            ext_t = cache["ext4"].to(device).float()  # [4,3,4] w2c
            contexts.append({
                "pair_id": f"probe{pi}",
                "scene": scene,
                "patch_grid": cache["patch_hw"],
                "teacher": {
                    "features": feats_t,
                    "feat_w": teacher_patch_conf(conf, cache["patch_hw"]),
                    "ext_w2c": ext_t,
                    "centers": _centers_from_ext(ext_t),
                    "depth": cache["depth4"][0].to(device).float(),
                    "valid": valid_mask_from_conf(conf[0]),
                },
                "images_meta": imgs4,  # CPU [4,3,H,W]; moved to device in forward_fn
            })
    return ProbeEvaluator(contexts, model=student)


def _probe_forward_factory(student, device: str):
    """forward_fn(images_meta, mask [S,ph,pw] bool) for ProbeEvaluator: zero-fill
    the masked 14x14 blocks of the 4 shared frames in normalized space (same
    convention as mask_image_blocks), run the student under bf16 autocast, and
    map outputs onto the ProbeEvaluator contract (post-head.norm feature dict,
    w2c ext, centers, depth). Grad-free: evaluate() wraps this in no_grad."""

    def forward_fn(images_meta, mask):
        images4 = images_meta.to(device).unsqueeze(0)  # [1,4,3,H,W]
        H, W = images4.shape[-2], images4.shape[-1]
        ph, pw = int(mask.shape[1]), int(mask.shape[2])
        m = mask.to(device).repeat_interleave(H // ph, dim=1).repeat_interleave(W // pw, dim=2)
        images4_in = images4 * (~m).unsqueeze(1).float()  # [S,1,H,W] broadcast over C
        tap, ext_w2c, depth_s = student_forward_c2m(student, images4_in, with_depth=True)
        head_norm = student.da3.model.head.norm
        features = {l: head_norm(tap[l].float()) for l in TAP_LAYERS}
        return {"features": features,
                "ext_w2c": ext_w2c,
                "centers": _centers_from_ext(ext_w2c),
                "depth": depth_s[0].float()}

    return forward_fn


# ---------------------------------------------------------------------------
# AB dual-teacher manifests + frozen reliability weights (layer 1+2)
# ---------------------------------------------------------------------------
def resolve_ab_manifest(path_tmpl: str, dataset: str, scene: str) -> str:
    """--v2_ab_manifest accepts a directory (<dir>/<ds>/<scene>.json), a
    template with {scene}/{ds}/{dataset}, or a plain file path."""
    if os.path.isdir(path_tmpl):
        p = os.path.join(path_tmpl, dataset, f"{scene}.json")
        if not os.path.exists(p):
            # builder sanitizes scene names: 20241230/828738/x -> 20241230__828738__x.json
            p = os.path.join(path_tmpl, dataset, f"{scene.replace('/', '__')}.json")
        return p
    if "{scene}" in path_tmpl or "{ds}" in path_tmpl or "{dataset}" in path_tmpl:
        return path_tmpl.format(scene=scene, ds=dataset, dataset=dataset)
    return path_tmpl


def load_ab_manifest(path: str, dataset: str, scene: str,
                     image_files: Sequence[str]) -> Dict:
    """Load + validate a per-scene AB manifest. Raises with an explicit message
    on any problem — NEVER silently falls back to on-the-fly sampling.

    Schema: compatible with the protocol JSON (N / teacher_N / n_shared /
    student_slots / eval_frames / image_files / ...). train_pairs items carry
    teacher_frames (context A), teacher_frames_B (context B) and student_frames;
    probe_pairs stay single-context (no B)."""
    if not os.path.exists(path):
        raise FileNotFoundError(            f"AB manifest not found: {path} (--v2_ab_manifest given; refusing to "
            f"fall back to on-the-fly protocol sampling)")
    with open(path) as f:
        proto = json.load(f)
    for key in ("N", "eval_frames", "train_pairs", "probe_pairs"):
        if key not in proto:
            raise ValueError(f"AB manifest {path}: missing required key '{key}'")
    if not isinstance(proto["train_pairs"], list) or not proto["train_pairs"]:
        raise ValueError(f"AB manifest {path}: train_pairs must be a non-empty list")
    if not isinstance(proto["probe_pairs"], list):
        raise ValueError(f"AB manifest {path}: probe_pairs must be a list")
    files = list(image_files)
    if "image_files" in proto:
        # rel-vs-abs path styles differ across invocation contexts; normalize
        norm = lambda ps: [os.path.abspath(str(p)) for p in ps]
        if norm(proto["image_files"]) != norm(files):
            raise ValueError(
                f"AB manifest {path}: image_files do not match the dataset's scene "
                f"image list (manifest {len(proto['image_files'])} vs scene {len(files)})")
    else:
        proto["image_files"] = files
    if int(proto["N"]) != len(files):
        raise ValueError(f"AB manifest {path}: N={proto['N']} != {len(files)} images")
    n_frames = len(files)
    for i, p in enumerate(proto["train_pairs"]):
        for key in ("teacher_frames", "teacher_frames_B", "student_frames"):
            if key not in p:
                raise ValueError(
                    f"AB manifest {path}: train_pairs[{i}] missing '{key}' "
                    f"(dual-context A/B protocol requires teacher_frames_B)")
        if len(p["teacher_frames_B"]) != len(p["teacher_frames"]):
            raise ValueError(
                f"AB manifest {path}: train_pairs[{i}] teacher_frames_B has "
                f"{len(p['teacher_frames_B'])} frames != teacher_frames "
                f"{len(p['teacher_frames'])} (same teacher_N required)")
        # A/B must share the SAME shared frames at the SAME slots, and the
        # student frames must equal those slots — reliability compares A/B
        # positionally; a violation would silently diff different frames.
        slots = [int(s) for s in proto.get("student_slots", [0, 2, 4, 6])]
        shared_a = [int(p["teacher_frames"][s]) for s in slots]
        shared_b = [int(p["teacher_frames_B"][s]) for s in slots]
        if shared_a != shared_b:
            raise ValueError(
                f"AB manifest {path}: train_pairs[{i}] shared slots {slots} differ "
                f"between A and B ({shared_a} vs {shared_b}) — A/B contexts must "
                f"differ ONLY in the extras")
        if shared_a != [int(x) for x in p["student_frames"]]:
            raise ValueError(
                f"AB manifest {path}: train_pairs[{i}] student_frames "
                f"{p['student_frames']} != teacher slots {slots} -> {shared_a}")
        for key in ("teacher_frames", "teacher_frames_B"):
            bad = [ix for ix in p[key] if not (0 <= int(ix) < n_frames)]
            if bad:
                raise ValueError(f"AB manifest {path}: train_pairs[{i}].{key} has "
                                 f"out-of-range indices {bad[:4]}...")
    for i, p in enumerate(proto["probe_pairs"]):
        for key in ("teacher_frames", "student_frames"):
            if key not in p:
                raise ValueError(f"AB manifest {path}: probe_pairs[{i}] missing '{key}'")
    proto.setdefault("dataset", dataset)
    proto.setdefault("scene", scene)
    proto.setdefault("strategy", "ab_manifest")
    proto.setdefault("tau", float("nan"))
    return proto


@torch.no_grad()
def compute_pair_reliability(cache_a: Dict, cache_b: Dict, head_norm,
                             device: str) -> Dict:
    """Frozen per-pair reliability weights from the two teacher contexts
    (A=teacher_frames, B=teacher_frames_B). Computed ONCE at cache time;
    student residuals are NEVER used.

    Feature level: uF[i,p] = 1 - cos(zA, zB) on the shared 4 views in the
    post-head.norm readout space (same taps / same norm as the distill loss),
    averaged over TAP_LAYERS -> uF [4,P].
    Edge level (6 shared edges, gauge-free via tta_v2.edges_from_w2c):
      uR_ij = angle(R_rel_A, R_rel_B) in degrees;
      uT_ij = 1 - cos(tdir_A, tdir_B), NaN where the teacher baseline is ~0
      (direction undefined) — dropped from the scene median.
    Scene medians tau_F/tau_R/tau_T and q = 1/(1+(u/tau)^2) are applied by the
    caller (they need ALL pairs). Returns u-tensors (CPU fp32) + per-pair
    geo_w = mean(qR) filled by the caller."""
    from free_geometry.tta_v2 import edges_from_w2c

    feats_a = {l: cache_a["feats"][l].to(device).float() for l in TAP_LAYERS}
    feats_b = {l: cache_b["feats"][l].to(device).float() for l in TAP_LAYERS}
    u_feat = 0.0
    for l in TAP_LAYERS:
        zA = head_norm(feats_a[l])
        zB = head_norm(feats_b[l])
        u_feat = u_feat + (1.0 - F.cosine_similarity(zA, zB, dim=-1))  # [1,4,P]
    u_feat = (u_feat / len(TAP_LAYERS))[0].detach().cpu()  # [4,P]

    ext_a = cache_a["ext4"].to(device).float()  # [4,3,4]
    ext_b = cache_b["ext4"].to(device).float()
    ea, eb = edges_from_w2c(ext_a), edges_from_w2c(ext_b)
    z = ((ea["R_rel"] - eb["R_rel"]) ** 2).sum(dim=(-2, -1))
    d = (z / 8.0).clamp(max=4.0).sqrt()  # |sin(phi/2)|, capped before asin
    u_rot = torch.rad2deg(2.0 * torch.asin(d.clamp(0.0, 1.0))).detach().cpu()  # [E]
    tn_a = F.normalize(ea["t_rel"], dim=-1, eps=1e-8)
    tn_b = F.normalize(eb["t_rel"], dim=-1, eps=1e-8)
    u_tdir = (1.0 - (tn_a * tn_b).sum(dim=-1)).detach().cpu()  # [E]
    near0 = ea["baseline"] < 1e-3 * ea["baseline"].mean()  # same convention as
    u_tdir = torch.where(near0.cpu(), torch.full_like(u_tdir, float("nan")), u_tdir)
    return {"u_feat": u_feat, "u_rot": u_rot, "u_tdir": u_tdir}


def finalize_reliability_(caches: List[Dict]) -> None:
    """Scene-level reliability: tau = median of u over ALL given pair caches,
    q = 1/(1+(u/tau)^2); stores q_feat/q_rot/q_tdir (CPU fp32) and geo_w
    (scalar float = mean(q_rot)) into each cache, in place."""
    u_feats = torch.cat([c["u_feat"].reshape(-1) for c in caches])
    u_rots = torch.cat([c["u_rot"].reshape(-1) for c in caches])
    u_tdirs = torch.cat([c["u_tdir"].reshape(-1) for c in caches])
    # floors: identical A/B contexts (or a scene where >50% of values are
    # exactly 0) would otherwise give tau=0 -> q=NaN. Same convention as
    # train_arms._q_from_u (max(tau, 1e-12)).
    tau_f = u_feats.median().clamp_min(1e-12)
    tau_r = u_rots.median().clamp_min(1e-12)
    finite_t = u_tdirs[torch.isfinite(u_tdirs)]
    tau_t = (finite_t.median() if finite_t.numel()
             else torch.tensor(1.0)).clamp_min(1e-12)
    for c in caches:
        qf = 1.0 / (1.0 + (c["u_feat"] / tau_f) ** 2)
        qr = 1.0 / (1.0 + (c["u_rot"] / tau_r) ** 2)
        qt = 1.0 / (1.0 + (c["u_tdir"] / tau_t) ** 2)  # NaN stays NaN
        c["q_feat"] = qf.float().cpu()
        c["q_rot"] = qr.float().cpu()
        c["q_tdir"] = qt.float().cpu()
        c["geo_w"] = float(qr.mean())


def compute_rkdc1h_loss(student, teacher_cache: Dict, images4_in: torch.Tensor,
                        patch_mask: torch.Tensor, rkd_weight: float = 1.5,
                        couple_weight: float = 1.0, rel_weight: float = 0.0,
                        v2_rot_w: Optional[float] = None,
                        v2_tdir_w: Optional[float] = None,
                        v2_couple_fix: bool = False,
                        v2_split: bool = False):
    """RKDC1H(+R) for DA3 = maskdistill + rkd_weight*rkd_huber + couple_weight*couple
    + rel_weight*rel_corrected (rel_weight>0 enables the corrected rel term).

    v2 (all default-off): v2_couple_fix swaps the couple term for the
    shared-valid-mask semantics of losses.loss_couple_centers; v2_rot_w /
    v2_tdir_w (>0) add the robust relative-pose branches rot_edges_huber /
    tdir_cos_loss with the teacher side detached, weighted independently
    (the caller applies the scene gate + ramp to these scalars); v2_split=True
    returns (base_total, extra, {"rot","tdir"}) with the branches UNWEIGHTED
    and NOT added, for --v2_grad_cap separate backward."""
    tap_feats_s, ext_s, depth_s = student_forward_c2m(student, images4_in, with_depth=True)
    device = images4_in.device
    cache = {
        "feats": {l: t.to(device) for l, t in teacher_cache["feats"].items()},
        "conf4": teacher_cache["conf4"].to(device),
        "patch_hw": teacher_cache["patch_hw"],
        "q_feat": (None if teacher_cache.get("q_feat") is None
                   else teacher_cache["q_feat"].to(device)),
    }
    head_norm = student.da3.model.head.norm
    feat_loss, feat_extra = loss_maskdistill(head_norm, cache, tap_feats_s, patch_mask,
                                             q_feat=cache["q_feat"])
    ext_t = teacher_cache["ext4"].to(device)
    rkd_loss, rkd_extra = loss_rkd_shared_pose_huber_w2c(ext_s, ext_t)
    if v2_couple_fix:
        cp_loss, cp_extra = _couple_fixed(ext_s, depth_s, ext_t,
                                          teacher_cache["depth4"].to(device),
                                          teacher_cache["conf4"].to(device))
    else:
        cp_loss, cp_extra = loss_couple_w2c(ext_s, depth_s, ext_t,
                                            teacher_cache["depth4"].to(device),
                                            teacher_cache["conf4"].to(device))
    extra = {**feat_extra, **rkd_extra, **cp_extra}
    # AB reliability (--v2_ab_manifest): frozen per-pair scalar downweights the
    # geometry terms. q_* live in the teacher cache; None when the flag is off.
    geo_w = teacher_cache.get("geo_w")
    if geo_w is not None:
        rkd_loss = rkd_loss * geo_w
        cp_loss = cp_loss * geo_w
        extra["geo_w"] = float(geo_w)
    total = feat_loss + rkd_weight * rkd_loss + couple_weight * cp_loss
    w_rot, w_tdir = (v2_rot_w or 0.0), (v2_tdir_w or 0.0)
    rel = None
    if w_rot > 0.0 or w_tdir > 0.0 or v2_split:
        rot_l, tdir_l, rel_extra = _v2_rel_branch(
            ext_s, ext_t, q_rot=teacher_cache.get("q_rot"),
            q_tdir=teacher_cache.get("q_tdir"))
        extra.update(rel_extra)
        if not v2_split:
            total = total + w_rot * rot_l + w_tdir * tdir_l
        else:
            rel = {"rot": rot_l, "tdir": tdir_l}
    if rel_weight > 0:
        rel_loss, rel_extra = loss_pose_rel(ext_s, ext_t)
        total = total + rel_weight * rel_loss
        extra.update(rel_extra)
    if v2_split:
        return total, extra, rel
    return total, extra


def compute_rkdc1hc_loss(student, teacher_cache: Dict, images4_in: torch.Tensor,
                         patch_mask: torch.Tensor, rkd_weight: float = 1.5,
                         couple_weight: float = 1.0, ctk_weight: float = 1.0,
                         v2_rot_w: Optional[float] = None,
                         v2_tdir_w: Optional[float] = None,
                         v2_couple_fix: bool = False,
                         v2_split: bool = False):
    """RKDC1HC = RKDC1H + ctk_weight * camera-token KD (direct pose-pathway
    supervision on the shared views' position-0 tokens, all 4 tap layers).
    v2 kwargs: same semantics as compute_rkdc1h_loss."""
    tap_feats_s, ext_s, depth_s, tap_cam_s = student_forward_c2m(
        student, images4_in, with_depth=True, with_cam=True)
    device = images4_in.device
    cache = {
        "feats": {l: t.to(device) for l, t in teacher_cache["feats"].items()},
        "conf4": teacher_cache["conf4"].to(device),
        "patch_hw": teacher_cache["patch_hw"],
        "q_feat": (None if teacher_cache.get("q_feat") is None
                   else teacher_cache["q_feat"].to(device)),
    }
    head_norm = student.da3.model.head.norm
    feat_loss, feat_extra = loss_maskdistill(head_norm, cache, tap_feats_s, patch_mask,
                                             q_feat=cache["q_feat"])
    ext_t = teacher_cache["ext4"].to(device)
    rkd_loss, rkd_extra = loss_rkd_shared_pose_huber_w2c(ext_s, ext_t)
    if v2_couple_fix:
        cp_loss, cp_extra = _couple_fixed(ext_s, depth_s, ext_t,
                                          teacher_cache["depth4"].to(device),
                                          teacher_cache["conf4"].to(device))
    else:
        cp_loss, cp_extra = loss_couple_w2c(ext_s, depth_s, ext_t,
                                            teacher_cache["depth4"].to(device),
                                            teacher_cache["conf4"].to(device))
    cam_t = {l: t.to(device) for l, t in teacher_cache["cam"].items()}
    ctk_loss, ctk_extra = loss_ctk(tap_cam_s, cam_t)
    extra = {**feat_extra, **rkd_extra, **cp_extra, **ctk_extra}
    geo_w = teacher_cache.get("geo_w")  # frozen AB reliability scalar (None = off)
    if geo_w is not None:
        rkd_loss = rkd_loss * geo_w
        cp_loss = cp_loss * geo_w
        extra["geo_w"] = float(geo_w)
    total = feat_loss + rkd_weight * rkd_loss + couple_weight * cp_loss \
        + ctk_weight * ctk_loss
    w_rot, w_tdir = (v2_rot_w or 0.0), (v2_tdir_w or 0.0)
    rel = None
    if w_rot > 0.0 or w_tdir > 0.0 or v2_split:
        rot_l, tdir_l, rel_extra = _v2_rel_branch(
            ext_s, ext_t, q_rot=teacher_cache.get("q_rot"),
            q_tdir=teacher_cache.get("q_tdir"))
        extra.update(rel_extra)
        if not v2_split:
            total = total + w_rot * rot_l + w_tdir * tdir_l
        else:
            rel = {"rot": rot_l, "tdir": tdir_l}
    if v2_split:
        return total, extra, rel
    return total, extra


def compute_c2m_loss(student, teacher_cache: Dict, images4_in: torch.Tensor, patch_mask: torch.Tensor,
                     pose_weight: float = 1.0, v2_rot_w: Optional[float] = None,
                     v2_tdir_w: Optional[float] = None, v2_split: bool = False):
    """C2M = maskdistill + pose_weight * rel-pose, the VGGT C2M_maskrel arm.
    pose_weight=0 recovers the protocol's pure-maskdistill fine variant
    (recommended for tau <= 0.55 datasets). Teacher cache tensors live on CPU
    and are moved to the GPU here per step. v2_rot_w/v2_tdir_w (>0) append the
    robust relative-pose branches (see compute_rkdc1h_loss); v2_split returns
    (base_total, extra, {"rot","tdir"}) for --v2_grad_cap."""
    tap_feats_s, ext_s, _ = student_forward_c2m(student, images4_in)
    device = images4_in.device
    cache = {
        "feats": {l: t.to(device) for l, t in teacher_cache["feats"].items()},
        "conf4": teacher_cache["conf4"].to(device),
        "patch_hw": teacher_cache["patch_hw"],
    }
    head_norm = student.da3.model.head.norm
    feat_loss, feat_extra = loss_maskdistill(head_norm, cache, tap_feats_s, patch_mask)
    ext_t = teacher_cache["ext4"].to(device)
    rel_loss, rel_extra = loss_pose_rel(ext_s, ext_t)
    total = feat_loss + pose_weight * rel_loss
    extra = {**feat_extra, **rel_extra, "rel": float(rel_loss)}
    w_rot, w_tdir = (v2_rot_w or 0.0), (v2_tdir_w or 0.0)
    rel = None
    if w_rot > 0.0 or w_tdir > 0.0 or v2_split:
        rot_l, tdir_l, v2_extra = _v2_rel_branch(ext_s, ext_t)
        extra.update(v2_extra)
        if not v2_split:
            total = total + w_rot * rot_l + w_tdir * tdir_l
        else:
            rel = {"rot": rot_l, "tdir": tdir_l}
    if v2_split:
        return total, extra, rel
    return total, extra


# ---------------------------------------------------------------------------
# Per-scene training (protocol section 3)
# ---------------------------------------------------------------------------
def draw_patch_mask(S: int, patch_hw, ratio: float, gen, device=None):
    """Same Bernoulli patch draw as mask_image_blocks, without touching pixels.
    Returns [1,S,P] float (1=masked). Shared by all mask_mode arms so the
    supervision/mask set Omega is IDENTICAL across the ablation."""
    ph, pw = patch_hw
    block = torch.rand(S, ph, pw, generator=gen, device=device) < ratio
    return block.reshape(1, S, ph * pw).float()


def install_token_mask(vit, pmask, where: str, mask_layer: int = 12):
    """Zero-fill masked PATCH tokens at an internal backbone point (hooks).

    where='shallow': right after patch projection, before any attention
      (MAE/iBOT-style entry mask, zero fill).
    where='feat':    at blocks[mask_layer] OUTPUT. DA3's cross-view (global)
      attention starts at block 13, so mask_layer=12 is the structural analog
      of "after the single-frame encoder, before the multi-view transformer".
    Patch tokens sit at [..., 1:, :] (position 0 is the camera token).
    Returns the hook handle (call .remove() after backward)."""
    mask_rows = pmask[0].bool()  # [S, P] (batch is always 1 -> S rows == B*S)

    def _hook(module, inp, out):
        if where == "shallow":          # out: (BS, P, C) patch-only
            out.masked_fill_(mask_rows.unsqueeze(-1), 0.0)
        else:                            # out: (BS, 1+P, C) camera at index 0
            out[:, 1:].masked_fill_(mask_rows.unsqueeze(-1), 0.0)
        return out

    target = vit.patch_embed if where == "shallow" else vit.blocks[mask_layer]
    return target.register_forward_hook(_hook)


@torch.no_grad()
def _unmasked_rot_median(student, probe_caches: List[Dict],
                         probe_images: List[torch.Tensor], device: str) -> float:
    """Scene-level rel-gate signal: with the student in its B=0 baseline state
    (right after reset_lora_, before any update), run each probe pair's shared
    4 frames UNMASKED through the student and measure the per-edge
    teacher-student relative-rotation angle (gauge-free, edges_from_w2c).
    Returns the median angle in degrees over all probe pairs' edges.
    Masked-probe-high + unmasked-low => occlusion-caused (learnable), gate
    passes; unmasked also high => untrustworthy/unlearnable teacher target,
    the caller disables the v2 rel branch for this scene."""
    from free_geometry.tta_v2 import edges_from_w2c

    angs = []
    for cache, imgs4 in zip(probe_caches, probe_images):
        images4 = imgs4.unsqueeze(0).to(device)  # clean input, no masking
        _, ext_s, _ = student_forward_c2m(student, images4)
        e_s = edges_from_w2c(ext_s)
        e_t = edges_from_w2c(cache["ext4"].to(device).float())
        z = ((e_s["R_rel"] - e_t["R_rel"]) ** 2).sum(dim=(-2, -1))
        d = (z / 8.0).sqrt().clamp(0.0, 1.0)  # |sin(phi/2)|
        angs.append(2.0 * torch.asin(d) * (180.0 / math.pi))
    return float(torch.cat(angs).median())


def train_scene_c2m(
    teacher: DepthAnything3,
    student,
    scene: str,
    image_files: Sequence[str],
    protocol: Dict,
    device: str = "cuda",
    steps: int = 100,
    epochs: int = EPOCHS,
    lr: float = LR,
    seed: int = 0,
    pose_weight: float = 1.0,
    arm: str = "c2m",
    mask_ratio: float = 0.5,
    mask_mode: str = "image",  # image | none | token_shallow | token_feat
    mask_layer: int = 12,
    mask_loss_positions: bool = True,
    ctk_weight: float = 1.0,
    rel_weight: float = 1.0,
    two_stage: float = 0.0,
    early_stop: bool = False,
    es_min_steps: int = 30,
    es_tol: float = 0.02,
    log_fn=print,
    trace_rows: Optional[list] = None,
    on_step=None,  # per-step callback(row: dict) for external loggers (swanlab)
    on_probe=None,  # per-probe-eval callback(record: dict), same record as the JSONL line
    v2: Optional[V2Config] = None,  # protocol-v2 knobs; None = pure v1 behavior
) -> Dict:
    """Run the fixed-100-step C2M adaptation for one scene. Returns stats.

    Device choreography (keeps peak memory low enough to share a GPU): the
    teacher is on `device` only while its features are cached, then goes back
    to CPU; the student moves to `device` for the training loop. Teacher caches
    and pair images live on CPU and are streamed per step.

    v2 (V2Config, all default-off): keeps the probe pairs' teacher caches for
    the GT-free probe, periodic LoRA checkpoints, the robust relative-pose
    branch, its gradient cap, and the fixed couple term. With v2=None (or every
    field default) the loss values are bit-identical to v1.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    all_pairs = protocol["train_pairs"] + protocol["probe_pairs"]
    n_train = len(protocol["train_pairs"])
    use_ab = bool(v2 and v2.ab)  # dual-teacher A/B contexts (--v2_ab_manifest)

    # ---- teacher caches (once per scene; zero teacher forwards in the loop) ----
    student.to("cpu")  # idle during caching; previous scene's LoRA is already saved
    teacher.to(device)
    caches, pair_images4 = [], []
    for pi, pair in enumerate(all_pairs):
        imgs = load_images_da3([image_files[i] for i in pair["teacher_frames"]])
        imgs = imgs.unsqueeze(0).to(device)  # [1,teacher_N,3,H,W] (teacher_N varies per pair in ratio_mix mode)
        ph, pw = imgs.shape[-2] // PATCH_SIZE, imgs.shape[-1] // PATCH_SIZE
        slots = list(range(0, 2 * len(pair["student_frames"]), 2))  # per-pair slots
        cache = cache_teacher_pair(teacher, imgs, slots, (ph, pw))
        if use_ab and pi < n_train:
            # context B: second frozen teacher forward, same teacher_N, no mask.
            if "teacher_frames_B" not in pair:
                raise ValueError(
                    f"v2.ab ({scene} train pair {pi}): missing 'teacher_frames_B' "
                    f"— the AB manifest must provide dual contexts for train pairs")
            imgs_b = load_images_da3([image_files[i] for i in pair["teacher_frames_B"]])
            imgs_b = imgs_b.unsqueeze(0).to(device)
            cache["B"] = cache_teacher_pair(
                teacher, imgs_b, slots,
                (imgs_b.shape[-2] // PATCH_SIZE, imgs_b.shape[-1] // PATCH_SIZE))
            del imgs_b
        caches.append(cache)
        pair_images4.append(imgs[0, slots].cpu().contiguous())  # [n_shared,3,H,W] CPU
        del imgs
    teacher.to("cpu")
    torch.cuda.empty_cache()
    patch_hw = caches[0]["patch_hw"]
    train_caches, train_images = caches[:n_train], pair_images4[:n_train]
    # v2: the probe pairs' caches used to be built and dropped; keep them.
    probe_caches, probe_images = caches[n_train:], pair_images4[n_train:]

    # ---- fresh LoRA (student == baseline at step 0) ----
    student.to(device)
    torch.manual_seed(stable_seed("lora_init", scene, seed))
    reset_lora_(student)
    params = student.get_trainable_params()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    n_steps = min(steps, n_train * epochs)
    # FIX 2026-09-17: schedule follows the ACTUAL step budget, not n_train*epochs.
    # Otherwise 20-pair runs (n_train*epochs=200, steps=100) only reach 64% of
    # the cosine at step 100, silently changing the LR curve with pair count.
    warm = max(1, round(n_steps * WARMUP_RATIO))
    # plateau checks must not fire during/right after the LR warmup ramp
    # (a 20-pair run warms up for 30 steps; comparing windows that still sit
    # inside the ramp falsely reads as convergence -> premature stop)
    es_min = max(es_min_steps, warm + 10)
    scheduler = SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warm),
         CosineAnnealingLR(optimizer, T_max=n_steps - warm, eta_min=1e-8)],
        [warm],
    )

    student.train()
    torch.cuda.reset_peak_memory_stats()
    step = 0
    losses = []
    stop = False

    # ---- protocol v2 state (all inert when v2 is None / defaults) ----
    v2_cf = bool(v2 and v2.couple_fix)
    v2_gR_hist: List[float] = []
    v2_C_R = None  # set after the first 10 updates: 4 * median(||g_aux||)
    # R/T split: per-branch weights fall back to --v2_rel_weight when unset.
    w_rot_base = w_tdir_base = 0.0
    if v2 is not None:
        w_rot_base = v2.rel_weight if v2.rot_weight is None else v2.rot_weight
        w_tdir_base = v2.rel_weight if v2.tdir_weight is None else v2.tdir_weight
    rel_requested = (w_rot_base > 0.0) or (w_tdir_base > 0.0)
    # scene-level rel gate: B=0 student, UNMASKED probe forwards. If even the
    # clean-input teacher-student rotation disagreement is huge, the rel target
    # is untrustworthy/unlearnable for this scene -> disable the rel branch.
    rel_gate = 1
    rot_deg_unmasked_median = None
    if rel_requested and v2 is not None and v2.rel_gate_deg > 0:
        rot_deg_unmasked_median = _unmasked_rot_median(
            student, probe_caches, probe_images, device)
        if rot_deg_unmasked_median > v2.rel_gate_deg:
            rel_gate = 0
            w_rot_base = 0.0
            w_tdir_base = 0.0
            log_fn(f"[{scene}] rel GATE TRIPPED: unmasked probe rot median "
                   f"{rot_deg_unmasked_median:.1f} deg > {v2.rel_gate_deg:.1f} "
                   f"-> v2 rel branch disabled for this scene (rel_gate=0)")
        else:
            log_fn(f"[{scene}] rel gate passed: unmasked probe rot median "
                   f"{rot_deg_unmasked_median:.1f} deg <= {v2.rel_gate_deg:.1f}")
    rel_active = (w_rot_base > 0.0) or (w_tdir_base > 0.0)
    use_v2_gc = bool(v2 and v2.grad_cap and rel_active)
    if v2 is not None and v2.grad_cap and not rel_requested:
        raise ValueError(
            "v2.grad_cap requires --v2_rel_weight > 0 or --v2_rot_weight / "
            "--v2_tdir_weight > 0")
    probe_evaluator = None
    probe_forward_fn = None
    probe_overlap = False
    q_summary = None
    if v2 is not None and v2.probe:
        probe_evaluator = _build_probe_evaluator(
            student, scene, protocol, probe_caches, probe_images, device)
        probe_forward_fn = _probe_forward_factory(student, device)
        # tiny scenes: the probe-pair fallback branch dedups against NOTHING,
        # so a probe pair can be identical to a train pair — warn, don't hide it
        _train_keys = {(tuple(sorted(p["teacher_frames"])), tuple(p["student_frames"]))
                       for p in protocol["train_pairs"]}
        probe_overlap = any(
            (tuple(sorted(p["teacher_frames"])), tuple(p["student_frames"])) in _train_keys
            for p in protocol["probe_pairs"])
        if probe_overlap:
            log_fn(f"[{scene}] WARNING: a probe pair is identical to a train pair "
                   f"(probe_train_overlap=true); the probe is not held out here")

    if use_ab:
        # layer 2: frozen reliability weights, ONCE, from the two teacher
        # contexts (student residuals never enter). Scene medians need all pairs.
        head_norm = student.da3.model.head.norm
        for cache in train_caches:
            cache.update(compute_pair_reliability(
                cache, cache["B"], head_norm, device))
        finalize_reliability_(train_caches)
        qf = torch.cat([c["q_feat"].reshape(-1) for c in train_caches]).numpy()
        qr = torch.cat([c["q_rot"].reshape(-1) for c in train_caches]).numpy()
        qt = torch.cat([c["q_tdir"].reshape(-1) for c in train_caches]).numpy()
        qt = qt[np.isfinite(qt)]
        q_summary = {
            "q_feat_mean": float(qf.mean()), "q_feat_q10": float(np.quantile(qf, 0.1)),
            "q_feat_q50": float(np.quantile(qf, 0.5)),
            "q_rot_mean": float(qr.mean()), "q_rot_q10": float(np.quantile(qr, 0.1)),
            "q_rot_q50": float(np.quantile(qr, 0.5)),
            "q_tdir_mean": float(qt.mean()) if qt.size else float("nan"),
            "q_tdir_q50": float(np.quantile(qt, 0.5)) if qt.size else float("nan"),
            "geo_w_mean": float(np.mean([c["geo_w"] for c in train_caches])),
        }
        log_fn(f"[{scene}] AB reliability: q_feat mean={q_summary['q_feat_mean']:.3f} "
               f"q10={q_summary['q_feat_q10']:.3f} | q_rot mean={q_summary['q_rot_mean']:.3f} "
               f"q10={q_summary['q_rot_q10']:.3f} | geo_w mean={q_summary['geo_w_mean']:.3f}")

    def _run_probe(step_n: int) -> None:
        res = probe_evaluator.evaluate(step_n, probe_forward_fn)
        rec = {"scene": scene, **res}
        if probe_overlap:
            rec["probe_train_overlap"] = True
        if rel_gate == 0:
            rec["rel_gate"] = 0  # explicit: rel branch disabled for this scene
        from free_geometry.tta_v2 import append_jsonl
        append_jsonl(os.path.join(v2.run_dir, "probe_trace", f"{scene}.jsonl"), rec)
        if on_probe is not None:
            try:
                on_probe(rec)
            except Exception as e:  # a logger must never kill training
                log_fn(f"[{scene}] WARNING: on_probe callback failed: {e}")

    stage_cut = round(steps * two_stage) if two_stage > 0 else None
    kinds = [p.get("kind", "fixed") for p in protocol["train_pairs"]]
    if stage_cut is not None:
        log_fn(f"[{scene}] two_stage={two_stage}: fixed-slot pairs for the first "
               f"{stage_cut} steps, then endpoint-anchored pairs "
               f"(fixed={kinds.count('fixed')}, se={kinds.count('se')}); "
               f"early_stop disabled (phase structure bounds training)")
    epoch = 0
    # step-0 baseline (student == frozen base right after the LoRA reset)
    if v2 is not None:
        if v2.ckpt:
            _save_v2_ckpt(student, v2.ckpt_dir, scene, 0)
        if v2.probe:
            _run_probe(0)
    while not stop:
        if stage_cut is not None:
            se_phase = step >= stage_cut
            pool = [i for i, k in enumerate(kinds) if (k == "se") == se_phase]
            if not pool:  # one phase exhausted its pair type; fall back to all
                pool = list(range(n_train))
        else:
            pool = list(range(n_train))
        order = [pool[i] for i in torch.randperm(
            len(pool), generator=torch.Generator().manual_seed(
                stable_seed("train_order", scene, epoch, seed))).tolist()]
        for pi in order:
            if stop:
                break
            cache = train_caches[pi]
            images4 = train_images[pi].unsqueeze(0).to(device)  # [1,4,3,H,W]
            gen = torch.Generator(device=images4.device).manual_seed(
                stable_seed("mask", scene, epoch, pi, seed))
            if mask_mode == "image":
                images4_in, pmask = mask_image_blocks(images4, mask_ratio, patch_hw, gen)
            else:
                # mask-position ablation: SAME Omega via the same seed, but the
                # input image stays clean; corruption happens inside the backbone
                # (token modes) or not at all (none).
                images4_in = images4
                pmask = draw_patch_mask(images4.shape[1], patch_hw, mask_ratio,
                                        gen, device=images4.device)
            # FIX 2026-09-17: separate the corruption mask (what gets hidden
            # from the student) from the loss mask (which positions are
            # supervised). Token hooks must always receive the 50% Omega,
            # regardless of the supervision mode.
            corruption_mask = pmask.clone()
            if not mask_loss_positions:
                # ablation (--loss_all_pos): supervise ALL patch positions
                pmask = torch.ones_like(pmask)
            hook = None
            if mask_mode in ("token_shallow", "token_feat"):
                vit = _unwrap_pretrained(student.da3.model.backbone)
                hook = install_token_mask(
                    vit, corruption_mask,
                    "shallow" if mask_mode == "token_shallow" else "feat",
                    mask_layer=mask_layer)
            torch.manual_seed(stable_seed("cf_rng", scene, epoch, pi, seed))
            # ramp: lambda(t) = w * min(1, t/20), t = completed updates (0-based)
            ramp = min(1.0, step / 20.0)
            w_rot_eff, w_tdir_eff = w_rot_base * ramp, w_tdir_base * ramp
            base_loss, rel = None, None
            if arm == "rkdc1hc":
                if use_v2_gc:
                    base_loss, extra, rel = compute_rkdc1hc_loss(
                        student, cache, images4_in, pmask, ctk_weight=ctk_weight,
                        v2_rot_w=w_rot_eff, v2_tdir_w=w_tdir_eff,
                        v2_couple_fix=v2_cf, v2_split=True)
                    loss = (base_loss if rel is None
                            else base_loss + w_rot_eff * rel["rot"] + w_tdir_eff * rel["tdir"])
                else:
                    loss, extra = compute_rkdc1hc_loss(
                        student, cache, images4_in, pmask, ctk_weight=ctk_weight,
                        v2_rot_w=w_rot_eff, v2_tdir_w=w_tdir_eff,
                        v2_couple_fix=v2_cf)
            elif arm == "rkdc1hr":
                loss, extra = compute_rkdc1h_loss(student, cache, images4_in, pmask,
                                                  rel_weight=rel_weight,
                                                  v2_couple_fix=v2_cf)
            elif arm == "rkdc1h":
                if use_v2_gc:
                    base_loss, extra, rel = compute_rkdc1h_loss(
                        student, cache, images4_in, pmask,
                        v2_rot_w=w_rot_eff, v2_tdir_w=w_tdir_eff,
                        v2_couple_fix=v2_cf, v2_split=True)
                    loss = (base_loss if rel is None
                            else base_loss + w_rot_eff * rel["rot"] + w_tdir_eff * rel["tdir"])
                else:
                    loss, extra = compute_rkdc1h_loss(
                        student, cache, images4_in, pmask,
                        v2_rot_w=w_rot_eff, v2_tdir_w=w_tdir_eff,
                        v2_couple_fix=v2_cf)
            else:
                if use_v2_gc:
                    base_loss, extra, rel = compute_c2m_loss(
                        student, cache, images4_in, pmask, pose_weight=pose_weight,
                        v2_rot_w=w_rot_eff, v2_tdir_w=w_tdir_eff, v2_split=True)
                    loss = (base_loss if rel is None
                            else base_loss + w_rot_eff * rel["rot"] + w_tdir_eff * rel["tdir"])
                else:
                    loss, extra = compute_c2m_loss(
                        student, cache, images4_in, pmask, pose_weight=pose_weight,
                        v2_rot_w=w_rot_eff, v2_tdir_w=w_tdir_eff)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at scene={scene} step={step + 1}: {float(loss)}")
            if use_ab:
                # frozen reliability diagnostics for this pair (constant per pair)
                extra.setdefault("geo_w", cache["geo_w"])
                extra["q_feat_mean"] = float(cache["q_feat"].mean())
                extra["q_rot_mean"] = float(cache["q_rot"].mean())
            g_base_norm = g_R_norm = g_rot_norm = g_tdir_norm = c_R = None
            if use_v2_gc:
                g_base = torch.autograd.grad(base_loss, params, allow_unused=True,
                                             retain_graph=True)
                g_base = [torch.zeros_like(p) if g is None else g
                          for p, g in zip(params, g_base)]
                g_base_norm = float(torch.norm(torch.stack([g.norm() for g in g_base])))
                if rel is not None and step < 10:
                    # calibration steps: THREE backwards (base/rot/tdir) so the
                    # per-branch gradient norms are visible in the trace
                    g_rot = torch.autograd.grad(rel["rot"], params, allow_unused=True,
                                                retain_graph=True)
                    g_tdir = torch.autograd.grad(rel["tdir"], params, allow_unused=True)
                    g_rot = [torch.zeros_like(p) if g is None else g
                             for p, g in zip(params, g_rot)]
                    g_tdir = [torch.zeros_like(p) if g is None else g
                              for p, g in zip(params, g_tdir)]
                    # merge by linearity: g_aux = w_rot*g_rot + w_tdir*g_tdir
                    g_aux = [w_rot_eff * a + w_tdir_eff * b
                             for a, b in zip(g_rot, g_tdir)]
                    g_rot = [w_rot_eff * g for g in g_rot]
                    g_tdir = [w_tdir_eff * g for g in g_tdir]
                    g_rot_norm = float(torch.norm(torch.stack([g.norm() for g in g_rot])))
                    g_tdir_norm = float(torch.norm(torch.stack([g.norm() for g in g_tdir])))
                elif rel is not None:
                    aux = w_rot_eff * rel["rot"] + w_tdir_eff * rel["tdir"]
                    g_aux = torch.autograd.grad(aux, params, allow_unused=True)
                    g_aux = [torch.zeros_like(p) if g is None else g
                             for p, g in zip(params, g_aux)]
                else:  # gate closed the rel branch: aux is identically zero
                    g_aux = [torch.zeros_like(p) for p in params]
                g_R_norm = float(torch.norm(torch.stack([g.norm() for g in g_aux])))
                if len(v2_gR_hist) < 10:
                    v2_gR_hist.append(g_R_norm)  # record only, no cap yet
                    scale_R, c_R = 1.0, float("nan")
                else:
                    if v2_C_R is None:
                        v2_C_R = 4.0 * float(np.median(v2_gR_hist))
                    scale_R = min(1.0, v2_C_R / max(g_R_norm, 1e-12))
                    c_R = v2_C_R
                with torch.no_grad():
                    for p, gb, ga in zip(params, g_base, g_aux):
                        p.grad = gb + scale_R * ga
                if hook is not None:
                    hook.remove()
                gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
            else:
                loss.backward()
                if hook is not None:
                    hook.remove()
                gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            losses.append(float(loss))
            if v2 is not None and step % v2.probe_every == 0 and (v2.probe or v2.ckpt):
                if v2.probe:
                    _run_probe(step)
                if v2.ckpt:
                    _save_v2_ckpt(student, v2.ckpt_dir, scene, step)
            row = {
                "scene": scene, "step": step, "epoch": epoch, "pair_idx": pi,
                "loss": float(loss), "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(gn),
                "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20,
                **extra,
            }
            if g_base_norm is not None:
                row["g_base_norm"] = g_base_norm
                row["g_R_norm"] = g_R_norm
                row["v2_C_R"] = c_R
                if g_rot_norm is not None:
                    row["g_rot_norm"] = g_rot_norm
                    row["g_tdir_norm"] = g_tdir_norm
            if rel_requested:
                row["rel_w_eff"] = w_rot_eff + w_tdir_eff
                row["w_rot_eff"] = w_rot_eff
                row["w_tdir_eff"] = w_tdir_eff
            if rot_deg_unmasked_median is not None:
                row["rel_gate"] = rel_gate
                row["rot_deg_unmasked_median"] = rot_deg_unmasked_median
            if trace_rows is not None:
                trace_rows.append(row)
            if on_step is not None:
                on_step(row)
            if step % 10 == 0 or step == 1:
                log_fn(
                    f"[{scene}] step {step}/{n_train * epochs} loss={float(loss):.4f} "
                    f"(md={extra['maskdistill']:.4f} rot={extra.get('rel_rot', 0.0):.4f} "
                    f"tdir={extra.get('rel_tdir', 0.0):.4f} rkd={extra.get('rkd_sh_d', 0.0) + extra.get('rkd_sh_a', 0.0):.4f} "
                    f"cp={extra.get('couple', 0.0):.4f} ctk={extra.get('ctk', 0.0):.4f}) lr={row['lr']:.2e} "
                    f"gn={row['grad_norm']:.3f} peak={row['peak_mem_mib']:.0f}MiB")
            if early_stop and stage_cut is None and step >= es_min and len(losses) >= 20:
                recent = float(np.mean(losses[-10:]))
                previous = float(np.mean(losses[-20:-10]))
                if previous - recent < es_tol * abs(previous):
                    log_fn(f"[{scene}] early-stop at step {step}: "
                           f"tail10={recent:.4f} vs prev10={previous:.4f} "
                           f"(improvement < {es_tol * 100:.0f}%)")
                    stop = True
                    break
            if step >= steps:
                stop = True
                break
        epoch += 1
        if stop:
            break
        if stage_cut is None and epoch >= epochs:
            break

    student.eval()
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    stats = {
        "scene": scene,
        "steps": step,
        "loss_first10": float(np.mean(losses[:10])),
        "loss_last10": float(np.mean(losses[-10:])),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "peak_mem_mib": peak_mib,
    }
    if q_summary is not None:
        stats["q_summary"] = q_summary
    log_fn(
        f"[{scene}] done: steps={step} loss {stats['loss_first10']:.4f} -> "
        f"{stats['loss_last10']:.4f} peak_mem={peak_mib:.0f}MiB")
    del caches, pair_images4, train_images, probe_caches, probe_images
    torch.cuda.empty_cache()
    return stats
