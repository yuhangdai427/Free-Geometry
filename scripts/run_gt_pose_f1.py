#!/usr/bin/env python3
"""GT-pose F1 for hiroom + scannetpp (runs in tmux, no timeout pressure)."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, 'src')
sys.path.insert(0, 'diagnostics/free_geometry')
import common as fg

# --- hiroom ---
fg.set_dataset('hiroom')
from depth_anything_3.bench.datasets.hiroom import HiRoomDataset
obj = HiRoomDataset()
recon = 'workspace/da3_164_maskrel_hiroom/recon'
base_f = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/hiroom_recon_baseline.json'))['scenes']
mr_s = json.load(open('workspace/da3_164_maskrel_hiroom/smoke_summary.json'))['scenes']

g_list, u_list, b_list = [], [], []
for npz_path in sorted(glob.glob(f'{recon}/*/*/exports/mini_npz/results.npz')):
    parts = npz_path.replace(recon + '/', '').split('/')
    scene = '/'.join(parts[:-3])
    if scene not in mr_s or scene not in base_f:
        continue
    exp_dir = os.path.dirname(os.path.dirname(npz_path))
    proto_tag = scene.replace('/', '__')
    pp = f'workspace/fgmig/protocols/hiroom/{proto_tag}.json'
    if not os.path.exists(pp):
        continue
    frames = json.load(open(pp))['eval_frames']
    data = fg.get_scene_data(scene)
    np.savez_compressed(os.path.join(exp_dir, 'gt_meta.npz'),
        extrinsics=np.asarray(data.extrinsics)[frames],
        intrinsics=np.asarray(data.intrinsics)[frames],
        image_files=np.array([data.image_files[i] for i in frames], dtype=object))
    fuse = os.path.join(exp_dir, 'fuse_gt_pose', 'pcd.ply')
    os.makedirs(os.path.dirname(fuse), exist_ok=True)
    try:
        obj.fuse3d(scene, npz_path, fuse, "recon_posed")
        r = obj.eval3d(scene, fuse)
        g_list.append(float(r['fscore']))
        u_list.append(mr_s[scene]['eval'].get('recon_fscore', 0))
        b_list.append(base_f[scene]['fscore'])
        print(f"  hr {scene:40s} gt_f1={g_list[-1]:.4f}", flush=True)
    except Exception as e:
        print(f"  hr {scene:40s} ERR {str(e)[:40]}", flush=True)

print(f"\nhiroom: base={np.mean(b_list):.4f} mr_pred={np.mean(u_list):.4f} "
      f"mr_gtpose={np.mean(g_list):.4f} (n={len(g_list)})", flush=True)

# --- scannetpp ---
fg.set_dataset('scannetpp')
from depth_anything_3.bench.datasets.scannetpp import ScanNetPP
obj2 = ScanNetPP()
recon2 = 'workspace/da3_164_maskrel_scannetpp/recon'
base_f2 = json.load(open('artifacts/diagnostics/final_protocol/da3_baseline/scannetpp_recon_baseline.json'))['scenes']
mr_s2 = json.load(open('workspace/da3_164_maskrel_scannetpp/smoke_summary.json'))['scenes']

g2, u2, b2 = [], [], []
for scene in sorted(os.listdir(recon2)):
    rp = os.path.join(recon2, scene, 'exports', 'mini_npz', 'results.npz')
    if not os.path.isfile(rp) or scene not in mr_s2 or scene not in base_f2:
        continue
    pp = f'workspace/fgmig/protocols/scannetpp/{scene}.json'
    if not os.path.exists(pp):
        continue
    frames = json.load(open(pp))['eval_frames']
    data = fg.get_scene_data(scene)
    exp = os.path.join(recon2, scene, 'exports')
    np.savez_compressed(os.path.join(exp, 'gt_meta.npz'),
        extrinsics=np.asarray(data.extrinsics)[frames],
        intrinsics=np.asarray(data.intrinsics)[frames],
        image_files=np.array([data.image_files[i] for i in frames], dtype=object))
    fuse = os.path.join(exp, 'fuse_gt_pose', 'pcd.ply')
    os.makedirs(os.path.dirname(fuse), exist_ok=True)
    try:
        obj2.fuse3d(scene, rp, fuse, "recon_posed")
        r = obj2.eval3d(scene, fuse)
        g2.append(float(r['fscore']))
        u2.append(mr_s2[scene]['eval'].get('recon_fscore', 0))
        b2.append(base_f2[scene]['fscore'])
        print(f"  sp {scene:14s} gt_f1={g2[-1]:.4f}", flush=True)
    except Exception as e:
        print(f"  sp {scene:14s} ERR {str(e)[:40]}", flush=True)

print(f"\nscannetpp: base={np.mean(b2):.4f} mr_pred={np.mean(u2):.4f} "
      f"mr_gtpose={np.mean(g2):.4f} (n={len(g2)})", flush=True)
print("\nDONE", flush=True)
