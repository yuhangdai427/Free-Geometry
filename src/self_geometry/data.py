"""Official DA3 loaders/evaluation, with immutable sampled RGB manifests."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from .common import digest, write_json


def tco_training_indices(name, count):
    """Official strides on supported datasets; explicit extension elsewhere."""
    if name in ('eth3d', 'dtu', '7scenes'):
        stride = 200 if name == '7scenes' else 5
        return list(range(0, count, stride)), dict(strategy='official_stride', stride=stride,
            note='TCO stride applied to the DA3 benchmark scene/sequence, not the upstream test split')
    if name in ('scannetpp', 'hiroom'):
        return np.linspace(0, count-1, min(count, 10), dtype=int).tolist(), dict(
            strategy='extension_uniform_10', note='Dataset absent from TCO; explicit sparse-view extension')
    raise ValueError(name)


def prepare_tco_training(name, scene, c, directory):
    """Select RGB before evaluation sampling; never provide GT to adaptation."""
    raw = dataset(name, c).get_data(scene)
    indices, policy = tco_training_indices(name, len(raw.image_files))
    if len(indices) < 2:
        raise ValueError('Official TCO stride selected fewer than two views; no silent sampling fallback')
    files = [raw.image_files[i] for i in indices]
    manifest = dict(dataset=name, scene=scene, role='tco_training', image_files=files,
        indices=indices, source_frames=len(raw.image_files), sampling=policy,
        images=[dict(path=p, bytes=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns) for p in files])
    manifest['fingerprint'] = digest(manifest)
    path = Path(directory)/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f'TCO training manifest changed: {path}')
    write_json(path, manifest)
    return manifest


def dataset(name, c):
    import depth_anything_3.bench.datasets
    from depth_anything_3.bench.registries import MV_REGISTRY
    ds = MV_REGISTRY.get(name)()
    root = Path(c['data_root'])
    ds.data_root = str(root/name)
    if name=='hiroom':
        ds.data_root = str(root/name/'data')
        ds.gt_root_path = str(root/name/'fused_pcd')
        ds.SCENES = (root/name/'selected_scene_list_val.txt').read_text().splitlines()
    return ds


def evaluation_data(ds, scene, manifest):
    """Reorder/subset all official per-view metadata to the immutable RGB manifest.

    DTU fusion reads get_data directly instead of exported gt_meta.npz. Bind its
    loader to this selection during evaluation so sampled smoke runs also align.
    """
    from addict import Dict
    data = ds.get_data(scene)
    lookup = {p: i for i, p in enumerate(data.image_files)}
    indices = [lookup[p] for p in manifest['image_files']]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate frame in RGB manifest')
    aux = {}
    for key, value in data.aux.items():
        if isinstance(value, (list, np.ndarray)) and len(value) == len(data.image_files):
            aux[key] = [value[i] for i in indices] if isinstance(value, list) else value[indices]
        else:
            aux[key] = value
    return Dict(image_files=list(manifest['image_files']),
                extrinsics=np.asarray(data.extrinsics)[indices],
                intrinsics=np.asarray(data.intrinsics)[indices], aux=Dict(aux))


def sampled_indices(count, maximum, seed):
    import random
    indices = list(range(count))
    if maximum > 0 and count > maximum:
        random.Random(seed).shuffle(indices)
        indices = sorted(indices[:maximum])
    return indices


def prepare(name, scene, c, directory, sampling_seed=42):
    from depth_anything_3.bench.evaluator import Evaluator
    ds = dataset(name,c)
    if scene not in ds.SCENES: raise ValueError(f'Unknown official scene {name}/{scene}')
    raw = ds.get_data(scene)
    if sampling_seed == 42:
        data = Evaluator._sample_frames(SimpleNamespace(max_frames=c['max_frames']), raw, scene)
    else:
        indices = sampled_indices(len(raw.image_files), c['max_frames'], sampling_seed)
        data = SimpleNamespace(image_files=[raw.image_files[i] for i in indices],
            extrinsics=np.asarray(raw.extrinsics)[indices], intrinsics=np.asarray(raw.intrinsics)[indices])
    files = list(data.image_files)
    manifest = dict(dataset=name,scene=scene,image_files=files,max_frames=c['max_frames'],sampling_seed=sampling_seed,
                    images=[{'path':p,'bytes':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in files])
    manifest['fingerprint']=digest(manifest)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f'Manifest mismatch: {path}; use a new output directory')
    write_json(path,manifest)
    # GT metadata is stored separately; training receives only RGB manifest.
    metadata = directory/'gt_meta.npz'
    temporary = directory/'gt_meta.tmp.npz'
    np.savez(temporary,extrinsics=np.asarray(data.extrinsics),intrinsics=np.asarray(data.intrinsics),image_files=np.asarray(files))
    temporary.replace(metadata)
    return manifest


def evaluate(name, scene, c, directory, stage, fuse_only=False, score_only=False):
    import fcntl
    directory=Path(directory)
    stage_dir=directory/stage
    stage_dir.mkdir(parents=True,exist_ok=True)
    with (stage_dir/'evaluation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _evaluate(name,scene,c,directory,stage,fuse_only,score_only)


def _evaluate(name, scene, c, directory, stage, fuse_only=False, score_only=False):
    import os
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(c['threads']))
    import random
    import torch
    import open3d as o3d
    from depth_anything_3.bench.evaluator import Evaluator
    from depth_anything_3.bench.utils import se3_to_relative_pose_error,calculate_auc_np,align_to_first_camera
    from depth_anything_3.utils.geometry import as_homogeneous
    directory=Path(directory); export=directory/stage/'exports'
    pred_path=export/'mini_npz/results.npz'
    metrics_path=directory/stage/'metrics.json'
    if metrics_path.exists():
        from .benchmark import metrics_for
        cached = json.loads(metrics_path.read_text())
        if not all(k in cached and np.isfinite(cached[k]) for k in metrics_for(name)):
            raise ValueError(f'Incomplete or nonfinite metric cache: {metrics_path}')
        return cached
    ds=dataset(name,c)
    with np.load(directory/'gt_meta.npz') as z: gt=dict(z)
    manifest=json.loads((directory/'manifest.json').read_text())
    fusion_path = directory/stage/'fusion.json'
    fusion_identity = digest(dict(manifest=manifest['fingerprint'],
                                  prediction_bytes=pred_path.stat().st_size,
                                  prediction_mtime=pred_path.stat().st_mtime_ns))
    fused = (fusion_path.exists() and
             json.loads(fusion_path.read_text()).get('identity') == fusion_identity and
             all((export/mode/'pcd.ply').exists() for mode in ('recon_unposed','recon_posed')))
    if score_only and not fused:
        raise ValueError('Run fuse successfully before score; fusion stamp or point clouds missing')
    if name == 'dtu':
        selected = evaluation_data(ds, scene, manifest)
        if not (np.array_equal(selected.extrinsics, gt['extrinsics']) and
                np.array_equal(selected.intrinsics, gt['intrinsics'])):
            raise ValueError('DTU camera metadata differs from prepared manifest')
        ds.get_data = lambda requested_scene: selected if requested_scene == scene else None
    if list(gt['image_files'])!=manifest['image_files']:
        raise ValueError('GT metadata image order differs from RGB manifest')
    with np.load(pred_path) as z:
        if len(z['depth'])!=len(manifest['image_files']):
            raise ValueError('Prediction frame count differs from RGB manifest')
    np.savez(export/'gt_meta.npz',**gt)
    metrics=dict(Evaluator._compute_pose_with_gt(None,str(pred_path),gt))
    # Official helper omits AUC@1; use its same relative error and histogram AUC.
    with np.load(pred_path) as z: ext=z['extrinsics']
    re,te=se3_to_relative_pose_error(align_to_first_camera(torch.from_numpy(as_homogeneous(ext))),align_to_first_camera(torch.from_numpy(as_homogeneous(gt['extrinsics']))),len(ext))
    metrics['auc01']=calculate_auc_np(re.numpy(),te.numpy(),max_threshold=1)[0]
    write_json(directory/stage/'pose_metrics.json', {k: float(v) for k,v in metrics.items()})
    for mode in ['recon_unposed','recon_posed']:
        random.seed(0);np.random.seed(0);torch.manual_seed(0);o3d.utility.random.seed(0)
        fuse=export/mode/'pcd.ply';fuse.parent.mkdir(exist_ok=True)
        mode_stamp = fuse.with_name('fusion.json')
        mode_fused = (fuse.exists() and mode_stamp.exists() and
                      json.loads(mode_stamp.read_text()).get('identity') == fusion_identity)
        if not score_only and not fused and not mode_fused:
            if name == 'eth3d':
                from .evaluation_stream import fuse_eth3d_streamed
                fuse_eth3d_streamed(ds, scene, str(pred_path), str(fuse), mode)
            else:
                ds.fuse3d(scene,str(pred_path),str(fuse),mode)
            write_json(mode_stamp, dict(identity=fusion_identity))
        if not fuse_only:
            # Degenerate-output tolerance (tco/vggt/dtu empty-cloud cells): keep
            # the pose scores and skip the failed recon mode instead of failing
            # the whole evaluation.
            try:
                metrics.update({mode+'_'+k:float(v) for k,v in ds.eval3d(scene,str(fuse)).items()})
            except Exception as exc:
                print(f'[evaluate] recon {mode} failed for {name}/{scene}: {exc}', flush=True)
    if not score_only:
        write_json(fusion_path, dict(identity=fusion_identity))
    if fuse_only:
        return metrics
    metrics={k:float(v) for k,v in metrics.items()}
    if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Nonfinite evaluation metric')
    write_json(metrics_path,metrics)
    return metrics
