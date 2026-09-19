import numpy as np

from free_geometry.evaluation import benchmark_arrays


def test_evaluation_undoes_crop_and_intrinsics_transform():
    # A 4x8 raw image was center cropped horizontally to 4x4.
    native = {
        "depth": np.ones((1, 4, 4), dtype=np.float32),
        "conf": np.ones((1, 4, 4), dtype=np.float32),
        "valid": np.ones((1, 4, 4), dtype=bool),
        "intrinsics": np.array([[[8, 0, 2], [0, 8, 2], [0, 0, 1]]], dtype=np.float32),
    }
    result = benchmark_arrays(
        native,
        [
            {
                "original_hw": [4, 8],
                "image_transform": [[1, 0, -2], [0, 1, 0], [0, 0, 1]],
            }
        ],
    )
    np.testing.assert_allclose(
        result["intrinsics"][0], [[4, 0, 2], [0, 8, 2], [0, 0, 1]]
    )
    assert not result["valid"][0, :, 0].any()
    assert result["valid"][0, :, 1:3].all()
    assert not result["valid"][0, :, 3].any()
    assert (result["depth"][~result["valid"]] == 0).all()
    assert native["valid"].all()  # Native exports remain unchanged.
