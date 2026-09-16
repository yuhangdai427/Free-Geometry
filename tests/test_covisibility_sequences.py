import json

import pytest

from vggt.vggt.test_time_adaption.covisibility_sequences import ResolvedScene, make_sequence


def test_sequence_rejects_scene_without_eight_unique_views():
    scene = ResolvedScene(
        dataset="eth3d", scene="unit", frame_indices=list(range(7)),
        frame_ids=[f"frame_{i}" for i in range(7)], image_files=[f"/tmp/{i}.jpg" for i in range(7)],
        extrinsics=None, intrinsics=None,
    )
    with pytest.raises(ValueError, match="cannot be constructed"):
        make_sequence(scene, seed=1, split="eval", sample_idx=0)


def test_sequence_is_deterministic_and_json_serializable():
    scene = ResolvedScene(
        dataset="scannetpp", scene="unit", frame_indices=list(range(9)),
        frame_ids=[f"frame_{i}" for i in range(9)], image_files=[f"/tmp/{i}.jpg" for i in range(9)],
        extrinsics=None, intrinsics=None,
    )
    first = make_sequence(scene, seed=43, split="eval", sample_idx=2)
    assert first == make_sequence(scene, seed=43, split="eval", sample_idx=2)
    assert len(set(first["eight_local_indices"])) == 8
    json.dumps(first)
