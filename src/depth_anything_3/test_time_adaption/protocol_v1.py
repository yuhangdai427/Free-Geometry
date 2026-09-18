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
import math
import os
import random
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """C2M maskdistill for DA3: post-LN (DualDPT shared norm) Huber+2cos on the
    4 tapped layers, teacher-conf weighted, masked positions only.

    teacher_cache["feats"][layer]: [1,4,P,3072] raw concat tokens of the shared
    views (bf16 cache, upcast here); tap_feats_s[layer]: [1,4,P,3072] student.
    """
    w = teacher_patch_conf(teacher_cache["conf4"], teacher_cache["patch_hw"]) * patch_mask
    w = w / w.mean().clamp_min(1e-8)
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
    """StudentModel with protocol LoRA: r=32/a=32/dropout=0 on ALL 40 aggregator
    blocks (attn qkv/proj + SwiGLU w12/w3), heads and camera token frozen."""
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
        lora_layers=list(range(DA3_NUM_LAYERS)),
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


def compute_rkdc1h_loss(student, teacher_cache: Dict, images4_in: torch.Tensor,
                        patch_mask: torch.Tensor, rkd_weight: float = 1.5,
                        couple_weight: float = 1.0, rel_weight: float = 0.0):
    """RKDC1H(+R) for DA3 = maskdistill + rkd_weight*rkd_huber + couple_weight*couple
    + rel_weight*rel_corrected (rel_weight>0 enables the corrected rel term)."""
    tap_feats_s, ext_s, depth_s = student_forward_c2m(student, images4_in, with_depth=True)
    device = images4_in.device
    cache = {
        "feats": {l: t.to(device) for l, t in teacher_cache["feats"].items()},
        "conf4": teacher_cache["conf4"].to(device),
        "patch_hw": teacher_cache["patch_hw"],
    }
    head_norm = student.da3.model.head.norm
    feat_loss, feat_extra = loss_maskdistill(head_norm, cache, tap_feats_s, patch_mask)
    ext_t = teacher_cache["ext4"].to(device)
    rkd_loss, rkd_extra = loss_rkd_shared_pose_huber_w2c(ext_s, ext_t)
    cp_loss, cp_extra = loss_couple_w2c(ext_s, depth_s, ext_t,
                                        teacher_cache["depth4"].to(device),
                                        teacher_cache["conf4"].to(device))
    total = feat_loss + rkd_weight * rkd_loss + couple_weight * cp_loss
    extra = {**feat_extra, **rkd_extra, **cp_extra}
    if rel_weight > 0:
        rel_loss, rel_extra = loss_pose_rel(ext_s, ext_t)
        total = total + rel_weight * rel_loss
        extra.update(rel_extra)
    return total, extra


def compute_rkdc1hc_loss(student, teacher_cache: Dict, images4_in: torch.Tensor,
                         patch_mask: torch.Tensor, rkd_weight: float = 1.5,
                         couple_weight: float = 1.0, ctk_weight: float = 1.0):
    """RKDC1HC = RKDC1H + ctk_weight * camera-token KD (direct pose-pathway
    supervision on the shared views' position-0 tokens, all 4 tap layers)."""
    tap_feats_s, ext_s, depth_s, tap_cam_s = student_forward_c2m(
        student, images4_in, with_depth=True, with_cam=True)
    device = images4_in.device
    cache = {
        "feats": {l: t.to(device) for l, t in teacher_cache["feats"].items()},
        "conf4": teacher_cache["conf4"].to(device),
        "patch_hw": teacher_cache["patch_hw"],
    }
    head_norm = student.da3.model.head.norm
    feat_loss, feat_extra = loss_maskdistill(head_norm, cache, tap_feats_s, patch_mask)
    ext_t = teacher_cache["ext4"].to(device)
    rkd_loss, rkd_extra = loss_rkd_shared_pose_huber_w2c(ext_s, ext_t)
    cp_loss, cp_extra = loss_couple_w2c(ext_s, depth_s, ext_t,
                                        teacher_cache["depth4"].to(device),
                                        teacher_cache["conf4"].to(device))
    cam_t = {l: t.to(device) for l, t in teacher_cache["cam"].items()}
    ctk_loss, ctk_extra = loss_ctk(tap_cam_s, cam_t)
    total = feat_loss + rkd_weight * rkd_loss + couple_weight * cp_loss \
        + ctk_weight * ctk_loss
    return total, {**feat_extra, **rkd_extra, **cp_extra, **ctk_extra}


def compute_c2m_loss(student, teacher_cache: Dict, images4_in: torch.Tensor, patch_mask: torch.Tensor,
                     pose_weight: float = 1.0):
    """C2M = maskdistill + pose_weight * rel-pose, the VGGT C2M_maskrel arm.
    pose_weight=0 recovers the protocol's pure-maskdistill fine variant
    (recommended for tau <= 0.55 datasets). Teacher cache tensors live on CPU
    and are moved to the GPU here per step."""
    tap_feats_s, ext_s, _ = student_forward_c2m(student, images4_in)
    device = images4_in.device
    cache = {
        "feats": {l: t.to(device) for l, t in teacher_cache["feats"].items()},
        "conf4": teacher_cache["conf4"].to(device),
        "patch_hw": teacher_cache["patch_hw"],
    }
    head_norm = student.da3.model.head.norm
    feat_loss, feat_extra = loss_maskdistill(head_norm, cache, tap_feats_s, patch_mask)
    rel_loss, rel_extra = loss_pose_rel(ext_s, teacher_cache["ext4"].to(device))
    return feat_loss + pose_weight * rel_loss, {**feat_extra, **rel_extra, "rel": float(rel_loss)}


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
) -> Dict:
    """Run the fixed-100-step C2M adaptation for one scene. Returns stats.

    Device choreography (keeps peak memory low enough to share a GPU): the
    teacher is on `device` only while its features are cached, then goes back
    to CPU; the student moves to `device` for the training loop. Teacher caches
    and pair images live on CPU and are streamed per step.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    all_pairs = protocol["train_pairs"] + protocol["probe_pairs"]

    # ---- teacher caches (once per scene; zero teacher forwards in the loop) ----
    student.to("cpu")  # idle during caching; previous scene's LoRA is already saved
    teacher.to(device)
    caches, pair_images4 = [], []
    for pair in all_pairs:
        imgs = load_images_da3([image_files[i] for i in pair["teacher_frames"]])
        imgs = imgs.unsqueeze(0).to(device)  # [1,teacher_N,3,H,W] (teacher_N varies per pair in ratio_mix mode)
        ph, pw = imgs.shape[-2] // PATCH_SIZE, imgs.shape[-1] // PATCH_SIZE
        slots = list(range(0, 2 * len(pair["student_frames"]), 2))  # per-pair slots
        caches.append(cache_teacher_pair(teacher, imgs, slots, (ph, pw)))
        pair_images4.append(imgs[0, slots].cpu().contiguous())  # [n_shared,3,H,W] CPU
        del imgs
    teacher.to("cpu")
    torch.cuda.empty_cache()
    patch_hw = caches[0]["patch_hw"]
    n_train = len(protocol["train_pairs"])
    train_caches, train_images = caches[:n_train], pair_images4[:n_train]

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
    stage_cut = round(steps * two_stage) if two_stage > 0 else None
    kinds = [p.get("kind", "fixed") for p in protocol["train_pairs"]]
    if stage_cut is not None:
        log_fn(f"[{scene}] two_stage={two_stage}: fixed-slot pairs for the first "
               f"{stage_cut} steps, then endpoint-anchored pairs "
               f"(fixed={kinds.count('fixed')}, se={kinds.count('se')}); "
               f"early_stop disabled (phase structure bounds training)")
    epoch = 0
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
            if arm == "rkdc1hc":
                loss, extra = compute_rkdc1hc_loss(student, cache, images4_in, pmask,
                                                   ctk_weight=ctk_weight)
            elif arm == "rkdc1hr":
                loss, extra = compute_rkdc1h_loss(student, cache, images4_in, pmask,
                                                  rel_weight=rel_weight)
            elif arm == "rkdc1h":
                loss, extra = compute_rkdc1h_loss(student, cache, images4_in, pmask)
            else:
                loss, extra = compute_c2m_loss(student, cache, images4_in, pmask,
                                               pose_weight=pose_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at scene={scene} step={step + 1}: {float(loss)}")
            loss.backward()
            if hook is not None:
                hook.remove()
            gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            losses.append(float(loss))
            row = {
                "scene": scene, "step": step, "epoch": epoch, "pair_idx": pi,
                "loss": float(loss), "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(gn),
                "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20,
                **extra,
            }
            if trace_rows is not None:
                trace_rows.append(row)
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
    log_fn(
        f"[{scene}] done: steps={step} loss {stats['loss_first10']:.4f} -> "
        f"{stats['loss_last10']:.4f} peak_mem={peak_mib:.0f}MiB")
    del caches, pair_images4, train_images
    torch.cuda.empty_cache()
    return stats
