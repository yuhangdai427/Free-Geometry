"""
VGGT Free-Geometry module.

This module provides components for Free-Geometry on a teacher/student VGGT pair.
"""

from .models import (
    VGGTFreeGeometryOutput,
    VGGTTeacherModel,
    VGGTStudentModel,
    count_parameters,
)
from .dataset import VGGTFreeGeometryDataset

__all__ = [
    "VGGTFreeGeometryOutput",
    "VGGTTeacherModel",
    "VGGTStudentModel",
    "VGGTFreeGeometryDataset",
    "count_parameters",
]
