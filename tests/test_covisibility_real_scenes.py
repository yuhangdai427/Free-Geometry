"""GPU integration checks against real benchmark frames, not synthetic geometry."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[1]
RUN_REAL = os.environ.get("RUN_COVISIBILITY_REAL_TESTS") == "1"
CASES = [
    ("7scenes", "chess"),
    ("scannetpp", "09c1414f1b"),
    ("eth3d", "courtyard"),
    ("hiroom", "20241230/828788/cam_sampled_13"),
]


@pytest.mark.skipif(not RUN_REAL, reason="requires the local benchmark datasets and GPU0")
@pytest.mark.parametrize(("dataset", "scene"), CASES)
def test_real_scene_matrix_and_nonoverlap_manifest(dataset, scene, tmp_path):
    output_root = tmp_path / "matrices"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run([
        sys.executable, "scripts/compute_covisibility_matrices.py",
        "--datasets", dataset, "--scenes", scene,
        "--output-root", str(output_root), "--device", "cuda:0",
    ], cwd=REPO_ROOT, env=environment, check=True)
    matrix_path = next((output_root / dataset).glob("*.npz"))
    with np.load(matrix_path) as data:
        overlap = data["max_directional_overlap"]
        threshold = float(data["adjacency_threshold"])
        assert overlap.shape[0] == overlap.shape[1] > 0
        assert np.allclose(overlap, overlap.T, atol=1e-6)
        assert np.all(np.diag(data["raw_covisibility"]) > 0)
    manifest = tmp_path / "selected.jsonl"
    subprocess.run([
        sys.executable, "scripts/select_nonoverlapping_views.py",
        "--matrix-root", str(output_root), "--output", str(manifest), "--datasets", dataset,
    ], cwd=REPO_ROOT, check=True)
    import json
    record = json.loads(manifest.read_text().strip())
    selected = record["selected_candidate_positions"]
    for index, other in enumerate(selected):
        for peer in selected[index + 1:]:
            assert overlap[other, peer] <= threshold + 1e-6
