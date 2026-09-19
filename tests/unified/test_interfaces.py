import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from free_geometry.adapters import REGISTRY, get_adapter
from free_geometry.config import ModelConfig


@pytest.mark.parametrize("name", list(REGISTRY))
def test_adapter_registry_is_lazy_and_complete(name):
    adapter = get_adapter(ModelConfig(name=name, weights="test-checkpoint"))
    for method in (
        "load",
        "reset_student",
        "prepare_images",
        "predict",
        "forward_teacher",
        "forward_student",
        "export_eval",
        "trainable_state",
        "load_trainable_state",
    ):
        assert callable(getattr(adapter, method))


def test_no_legacy_core_imports():
    root = Path(__file__).resolve().parents[2] / "src/free_geometry"
    for p in root.rglob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    x in (node.module or "")
                    for x in ("diagnostics", "protocol_v1", "tta_v2")
                )


def test_installed_cli(tmp_path):
    from PIL import Image

    for i in range(13):
        Image.new("RGB", (28, 28), (i * 10, 40, 70)).save(tmp_path / f"{i}.png")
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "free_geometry.cli",
            "prepare",
            "--data-root",
            str(tmp_path),
            "--scene",
            "test",
            "--output",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "free_geometry.cli", "inspect", str(manifest)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["unique_train_shared"] == 10
