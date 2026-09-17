"""Adapter interface: the ONLY model-specific layer of the unified TTA protocol.

An adapter provides, per model:
  load(device)                     — frozen teacher (original checkpoint)
  reset_student(device)            — fresh trainable student (LoRA re-init or
                                     full-model reload for full-FT models)
  student_params()                 — trainable parameter list for the optimizer
  student_label()                  — e.g. "lora_r32" / "fullft_117M"
  forward_teacher(images_N, slots) — cache dict on shared slots:
      readouts {name: [1,S,P,C]}   — native readout space, patch tokens only
      depth [S,H,W], conf [S,H,W]  — teacher depth + confidence
      centers [S,3]                — camera centers (any gauge, teacher's own)
      valid [S,H,W] bool           — teacher-derived valid-pixel mask
  forward_student(images_M)        — SAME keys, plus differentiable centers
  export_eval(images_eval)         — {depth [N,H,W], conf, extr_w2c [N,4,4],
                                     intr [N,3,3]} for the metric chain
  patch_hw                         — (ph, pw) of the last forward
Convention: centers live in the model's own world frame on BOTH sides; the
losses are gauge-free so no cross-model alignment is needed.
"""
from typing import Dict, List, Tuple


class BaseAdapter:
    model_key: str = "base"
    uses_lora: bool = True

    def load(self, device="cuda"):
        raise NotImplementedError

    def reset_student(self, device="cuda"):
        raise NotImplementedError

    def student_params(self) -> List:
        raise NotImplementedError

    def student_label(self) -> str:
        raise NotImplementedError

    def forward_teacher(self, images_N, shared_slots: List[int]) -> Dict:
        raise NotImplementedError

    def forward_student(self, images_M) -> Dict:
        raise NotImplementedError

    def export_eval(self, images_eval) -> Dict:
        raise NotImplementedError

    @property
    def patch_hw(self) -> Tuple[int, int]:
        raise NotImplementedError
