"""ETH3D's original TSDF operations with one original-resolution RGBD at a time."""
import json
from pathlib import Path
import time
import cv2
import numpy as np


def eth3d_frame(ds, scene, image_path, depth, intrinsics, scale, mode, image_shape):
    h, w = image_shape
    model_h, model_w = depth.shape
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = ds._load_gt_mask(scene, Path(image_path).name, (h, w))
    depth = ds._mask_invalid_depth(depth, mask) * scale
    intrinsics = intrinsics.copy()
    if mode == 'recon_unposed':
        intrinsics[0, :] *= w/model_w
        intrinsics[1, :] *= h/model_h
    return depth, intrinsics


def fuse_eth3d_streamed(ds, scene, result_path, fuse_path, mode):
    import open3d as o3d
    from addict import Dict
    from depth_anything_3.bench.utils import create_tsdf_volume, sample_points_from_mesh
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    if mode not in ('recon_posed', 'recon_unposed'):
        raise ValueError(mode)
    gt = ds._load_gt_meta(result_path)
    if gt is None:
        gt = ds.get_data(scene)
    with np.load(result_path) as z:
        pred = Dict({k: z[k] for k in ('depth', 'extrinsics', 'intrinsics')})
    _, _, scale, aligned = align_poses_umeyama(gt.extrinsics.copy(), pred.extrinsics.copy(),
                                              return_aligned=True, ransac=True, random_state=42)
    extrinsics = aligned if mode == 'recon_unposed' else gt.extrinsics
    intrinsics = pred.intrinsics if mode == 'recon_unposed' else gt.intrinsics
    volume = create_tsdf_volume(voxel_length=ds.voxel_length, sdf_trunc=ds.sdf_trunc)
    tick = time.monotonic()
    for i, path in enumerate(gt.image_files):
        rgb = cv2.imread(str(path))
        if rgb is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth, k = eth3d_frame(ds, scene, path, pred.depth[i], intrinsics[i], scale, mode, rgb.shape[:2])
        h, w = depth.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)), o3d.geometry.Image(depth.astype(np.float32)),
            depth_trunc=ds.max_depth, convert_rgb_to_intensity=False, depth_scale=1.)
        camera = o3d.camera.PinholeCameraIntrinsic(w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        volume.integrate(rgbd, camera, extrinsics[i])
        print(json.dumps(dict(event='eth3d_tsdf_frame', mode=mode, frame=i+1,
                              frames=len(gt.image_files), seconds=time.monotonic()-tick)), flush=True)
        del rgb, depth, rgbd
    print(json.dumps(dict(event='eth3d_tsdf_mesh', mode=mode)), flush=True)
    mesh = volume.extract_triangle_mesh()
    del volume
    pcd = sample_points_from_mesh(mesh, ds.sampling_number)
    Path(fuse_path).parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(fuse_path), pcd):
        raise OSError(f'Cannot save {fuse_path}')
