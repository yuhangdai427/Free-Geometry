"""GT-free probe evaluation (protocol v2 core).

Owns mask generation + robust-loss composition ONLY. The model forward is
injected per pipeline: forward_fn(images_meta, mask) -> student outputs, so
each pipeline keeps its own pixel-fill convention (DA3: zero-fill in
normalized space; free_geometry: ImageNet-mean fill in [0,1] space) — the
probe never touches pixels.
"""
import hashlib
import json
import os
from typing import Callable, Dict, List, Optional

import torch

from ..losses import loss_rkd_centers
from .robust_losses import (couple_robust, edges_from_w2c, huber_cos_weighted,
                            rot_edges_huber)


def stable_seed(*parts) -> int:
    """Deterministic seed from arbitrary parts: sha256(key)[:8] as big-endian
    int, key = "::".join(str(p)).

    Byte-identical to depth_anything_3.test_time_adaption.protocol_v1.stable_seed
    (and diagnostics/free_geometry/common.stable_seed); defined here so the
    free_geometry core stays importable without the DA3 chain (cf. trainer.py).
    """
    key = "::".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def make_patch_mask(S: int, ph: int, pw: int, ratio: float, seed: int) -> torch.Tensor:
    """Independent per-patch Bernoulli mask, [S,ph,pw] bool (True = masked).

    Drawn on a dedicated CPU generator so the global RNG state is untouched;
    the same seed reproduces the identical mask across processes and time.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.rand(S, ph, pw, generator=gen) < ratio


def append_jsonl(path: str, record: Dict) -> None:
    """Append one JSON-serializable record as a single JSONL line, creating
    parent directories as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class ProbeEvaluator:
    """Fixed masked re-feeds of the probe contexts, scored by robust loss
    components against the frozen teacher.

    contexts: list of dicts:
      pair_id      str, identifies the context in records
      scene        str, mask seeding only (defaults to pair_id)
      patch_grid   (ph, pw) patch grid the mask is drawn on
      mask_ratio   optional, default 0.5
      teacher      frozen quantities (probe detaches what it uses):
        features   Tensor [...,C] or {name: Tensor} readouts
        feat_w     [leading dims of features] conf weight (mean-1 style)
        ext_w2c    [S,3,4] or [1,S,3,4]
        centers    [S,3]
        depth      [S,H,W]
        valid      [S,H,W] bool
      images_meta  opaque handle forwarded to forward_fn as-is
    forward_fn(images_meta, mask [S,ph,pw] bool) -> dict with keys "features"
    (Tensor or dict), "ext_w2c", "centers", "depth". Teacher tensors and
    student outputs must already live on the same device.

    The two masks per context (k in {0,1}, seed stable_seed("probemask",
    scene, pair_id, k)) are drawn at construction and cached: identical across
    evaluate() calls and across steps. evaluate() runs fully under
    torch.no_grad(); if a model was passed, its training mode is saved, set to
    eval, and restored afterwards — forward_fn must not depend on grad state
    and must not flip the model mode itself.
    """

    def __init__(self, contexts: List[Dict], model: Optional[torch.nn.Module] = None):
        self.model = model
        self._units = []
        for ctx in contexts:
            ext_t = ctx["teacher"]["ext_w2c"]
            S = (ext_t[0] if ext_t.dim() == 4 else ext_t).shape[0]
            ph, pw = ctx["patch_grid"]
            scene = ctx.get("scene", ctx["pair_id"])
            masks = []
            for k in (0, 1):
                seed = stable_seed("probemask", scene, ctx["pair_id"], k)
                masks.append((k, make_patch_mask(S, ph, pw, ctx.get("mask_ratio", 0.5), seed)))
            self._units.append({"ctx": ctx, "teacher": ctx["teacher"],
                                "R_rel_t": edges_from_w2c(ext_t)["R_rel"].detach(),
                                "masks": masks})

    def evaluate(self, step: int, forward_fn: Callable) -> Dict:
        """Score every context x fixed mask via forward_fn. Returns
        {"step", "records": [{pair_id, mask_id, components, total}]} with
        components {"feature", "rot_deg", "rkd", "couple"} and
        total = feature + 1.5*rkd + couple + rot_deg — a raw unnormalized
        sum recorded for inspection only; checkpoint selection must run the
        controller on the per-component traces, never on total.
        """
        was_training = None
        if self.model is not None:
            was_training = self.model.training
            self.model.eval()
        try:
            with torch.no_grad():
                records = []
                for unit in self._units:
                    for mask_id, mask in unit["masks"]:
                        out = forward_fn(unit["ctx"]["images_meta"], mask)
                        records.append(self._score(unit, out, mask_id))
        finally:
            if was_training:
                self.model.train()
        return {"step": int(step), "records": records}

    def _score(self, unit: Dict, out: Dict, mask_id: int) -> Dict:
        ctx, teacher = unit["ctx"], unit["teacher"]
        feat = self._feature_loss(out["features"], teacher)
        edges_s = edges_from_w2c(out["ext_w2c"])
        if edges_s["R_rel"].shape[0]:
            rot = rot_edges_huber(edges_s["R_rel"], unit["R_rel_t"])
            rot_deg = rot["angle_deg"].mean()
        else:
            rot_deg = edges_s["baseline"].sum() * 0.0
        rkd = loss_rkd_centers(out["centers"], teacher["centers"])
        cp = couple_robust(out["centers"], teacher["centers"].detach(),
                           out["depth"], teacher["depth"].detach(),
                           teacher["valid"], huber_delta=None)
        components = {"feature": float(feat), "rot_deg": float(rot_deg),
                      "rkd": float(rkd), "couple": 0.0}
        record = {"pair_id": ctx["pair_id"], "mask_id": mask_id,
                  "components": components}
        if cp["skipped"]:
            record["couple_skipped"] = cp["reason"]
        else:
            components["couple"] = float(cp["loss"])
        record["total"] = float(components["feature"] + 1.5 * components["rkd"]
                                + components["couple"] + components["rot_deg"])
        return record

    @staticmethod
    def _feature_loss(out_feat, teacher) -> torch.Tensor:
        w = teacher["feat_w"]
        if isinstance(out_feat, dict):
            names = list(out_feat.keys())
            total = 0.0
            for name in names:
                total = total + huber_cos_weighted(
                    out_feat[name].float(), teacher["features"][name].float().detach(), w)
            return total / max(1, len(names))
        return huber_cos_weighted(out_feat.float(), teacher["features"].float().detach(), w)
