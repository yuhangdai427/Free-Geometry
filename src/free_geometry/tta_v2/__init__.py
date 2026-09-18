"""Protocol-v2 shared TTA core: robust losses, GT-free probe evaluation and
checkpoint selection/fallback, consumed by both TTA training pipelines (DA3
test_time_adaption, VGGT train_arms). Each pipeline injects its own model
forward and pixel-fill convention — nothing here touches pixels or models.
"""
from .controller import ControllerConfig, rel_change, select, should_stop
from .probe import (ProbeEvaluator, append_jsonl, make_patch_mask, stable_seed)
from .robust_losses import (couple_robust, edges_from_w2c, huber_cos_weighted,
                            rot_edges_huber, tdir_cos_loss)

__all__ = [
    "ControllerConfig", "rel_change", "select", "should_stop",
    "ProbeEvaluator", "append_jsonl", "make_patch_mask", "stable_seed",
    "couple_robust", "edges_from_w2c", "huber_cos_weighted",
    "rot_edges_huber", "tdir_cos_loss",
]
