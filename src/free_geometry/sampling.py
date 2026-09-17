"""Frame selection: thin wrapper reusing the frozen DA3 protocol builder.

Same tau-dispatched GT-free selection as all existing results (dense equidistant
SIFT vs random), 8-view teacher with 4 shared student frames at slots [0,2,4,6].
The new-model pipelines import from here so selection NEVER diverges.
"""
import sys
from typing import List, Sequence

# Reuse the audited implementation verbatim.
sys.path.insert(0, __file__.rsplit("/src/free_geometry/", 1)[0] + "/src")
from depth_anything_3.test_time_adaption import protocol_v1 as _P  # noqa: E402

TEACHER_N = 8
N_SHARED = 4


def build_protocol(image_files: Sequence[str], scene: str, dataset: str,
                   n_train: int = 10, seed_tag: str = "fgmig"):
    """GT-free per-scene protocol: 10 train + 2 probe 8:4 pairs + eval frames."""
    return _P.build_scene_protocol(image_files, scene, dataset=dataset,
                                   n_train=n_train, n_probe=2,
                                   n_shared=N_SHARED, teacher_N=TEACHER_N)


def student_slots(n_shared: int = N_SHARED) -> List[int]:
    return list(range(0, 2 * n_shared, 2))
