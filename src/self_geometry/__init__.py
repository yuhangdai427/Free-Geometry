"""Independent Self-Geometry reproduction; not the authors' implementation."""
from pathlib import Path
import sys
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "vendor/vggt", ROOT / "vendor/da3", ROOT / "vendor/LightGlue"):
    sys.path.insert(0, str(directory))
