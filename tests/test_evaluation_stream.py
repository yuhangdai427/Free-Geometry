import numpy as np
import pytest
from addict import Dict
from self_geometry.evaluation_stream import eth3d_frame
from depth_anything_3.bench.datasets import eth3d


@pytest.mark.parametrize('mode', ['recon_posed', 'recon_unposed'])
def test_streamed_rgbd_matches_original_preparation(monkeypatch, mode):
    ds = eth3d.ETH3D()
    rng = np.random.default_rng(9)
    pred = Dict(depth=rng.uniform(.2, 8, (3, 6, 8)).astype(np.float32),
                intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0))
    pred.depth[0, 0, 0] = np.nan
    gt = Dict(image_files=['a.png', 'b.png', 'c.png'],
              intrinsics=pred.intrinsics*2, extrinsics=pred.extrinsics.copy())
    scale = 3.25
    monkeypatch.setattr(eth3d, 'align_poses_umeyama', lambda *a, **kw: (None, None, scale, pred.extrinsics))
    mask = np.ones((12, 16), dtype=bool); mask[:, :3] = False
    ds._load_gt_mask = lambda *args: mask
    original = ds._prep_posed if mode == 'recon_posed' else ds._prep_unposed
    depths, intrinsics, _ = original(pred, gt, [(12, 16)]*3, scene='test')
    source = pred.intrinsics if mode == 'recon_unposed' else gt.intrinsics
    for i in range(3):
        depth, intr = eth3d_frame(ds, 'test', gt.image_files[i], pred.depth[i], source[i], scale, mode, (12, 16))
        np.testing.assert_array_equal(depth, depths[i])
        np.testing.assert_array_equal(intr, intrinsics[i])
