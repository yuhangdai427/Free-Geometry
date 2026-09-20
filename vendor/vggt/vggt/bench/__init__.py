"""
VGGT Benchmark Evaluation Module.

This module provides evaluation infrastructure for VGGT models,
reusing generic utilities from DA3 where possible.
"""

from .evaluator import VGGTEvaluator
from .registries import VGGT_MV_REGISTRY

__all__ = [
    "VGGTEvaluator",
    "VGGT_MV_REGISTRY",
]
