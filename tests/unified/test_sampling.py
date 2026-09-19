import subprocess
import sys

import pytest
from PIL import Image

from free_geometry.config import SamplingConfig, from_dict
from free_geometry.sampling import (
    SceneSource,
    build_manifest,
    fingerprint,
    sample_tasks,
    validate_manifest,
)


@pytest.mark.parametrize("n", list(range(8, 21)) + [49, 50, 100, 1000])
@pytest.mark.parametrize("ratio", [(8, 4), (7, 3), (6, 2)])
def test_tasks(n, ratio):
    t, s = ratio
    for seed in range(100):
        c = SamplingConfig(teacher_frames=t, student_frames=s, seed=seed)
        train, probe = sample_tasks(n, c, dense=n >= 50)
        assert len({tuple(p["shared"]) for p in train + probe}) == 12
        for p in train:
            for side in ("a", "b"):
                frames = p["teacher_" + side]
                assert len(set(frames)) == t
                assert [frames[i] for i in p["slots_" + side]] == p["shared"]
            assert p["extras_overlap"] == max(0, 2 * t - s - n)
        assert sample_tasks(n, c, dense=n >= 50) == (train, probe)


def test_small_scenes_never_call_sift_and_bad_files(tmp_path):
    files = []
    for i in range(50):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        files.append(str(p))
    calls = []

    def tau(paths):
        calls.append(len(paths))
        return 0.6

    source = SceneSource("s", files[:49], list(range(49)))
    m = build_manifest(source, tau_fn=tau)
    assert not calls and m["strategy"] == "random"
    assert (
        build_manifest(source, SamplingConfig(dense_min_frames=1), tau_fn=tau)[
            "strategy"
        ]
        == "random"
    )
    assert not calls
    source = SceneSource("s", files, list(range(50)))
    assert not build_manifest(source, SamplingConfig(sift_mode="off"), tau_fn=tau)[
        "sift_called"
    ]
    assert not calls
    assert build_manifest(source, tau_fn=tau)["strategy"] == "random"
    assert calls == [50]
    m = build_manifest(source, tau_fn=lambda _: 0.60001)
    assert m["strategy"] == "dense"
    m["train"][0]["slots_a"][0] = 1
    m["fingerprint"] = fingerprint(m)
    with pytest.raises(ValueError, match="mapping"):
        validate_manifest(m)
    (tmp_path / "bad.png").write_text("bad")
    source.image_files.append(str(tmp_path / "bad.png"))
    source.frame_ids.append(50)
    assert len(build_manifest(source, tau_fn=tau)["bad_files"]) == 1


def test_insufficient_and_config(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        sample_tasks(3, SamplingConfig(teacher_frames=3, student_frames=2))
    with pytest.raises(ValueError):
        from_dict({"loss": {"unknown": 1}})
    with pytest.raises(ValueError):
        from_dict({"train": {"max_steps": True}})
    with pytest.raises(ValueError):
        from_dict({"train": {"lr": float("nan")}})
    script = 'import free_geometry.cli, sys; assert "torch" not in sys.modules; assert "depth_anything_3" not in sys.modules'
    subprocess.run([sys.executable, "-c", script], check=True)
