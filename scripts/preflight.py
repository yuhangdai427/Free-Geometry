#!/usr/bin/env python3
"""Check existing RGB, cameras and GT assets for all 90 official scenes."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_geometry import ROOT
from self_geometry.benchmark import DATASETS, SCENE_COUNTS
from self_geometry.common import config, write_json
from self_geometry.data import dataset
from depth_anything_3.bench.evaluator import Evaluator


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/preflight.json')
    a = p.parse_args(argv)
    c, records, failures = config(a.config, a.set), [], []
    for name in DATASETS:
        ds = dataset(name, c)
        if len(ds.SCENES) != SCENE_COUNTS[name]:
            failures.append(dict(dataset=name, error=f'Expected {SCENE_COUNTS[name]} scenes, found {len(ds.SCENES)}'))
        for scene in ds.SCENES:
            try:
                data = ds.get_data(scene)
                selected = Evaluator._sample_frames(SimpleNamespace(max_frames=c['max_frames']), data, scene)
                files = list(selected.image_files)
                if not (len(files) >= 2 and len(files) == len(selected.extrinsics) == len(selected.intrinsics)):
                    raise ValueError('Frame/camera count mismatch')
                assets = list(files)
                for key in ('gt_mesh_path', 'gt_pcd_path'):
                    if key in selected.aux:
                        assets.append(selected.aux[key])
                if name == 'dtu':
                    assets += list(selected.aux.mask_files)
                    scan = int(scene[4:])
                    assets += [ds.get_3dgtpath(scene),
                               str(Path(ds.data_root) / f'SampleSet/mvs_data/ObsMask/ObsMask{scan}_10.mat'),
                               str(Path(ds.data_root) / f'SampleSet/mvs_data/ObsMask/Plane{scan}.mat')]
                for path in assets:
                    if not Path(path).is_file() or not Path(path).stat().st_size:
                        raise FileNotFoundError(path)
                records.append(dict(dataset=name, scene=scene, available_frames=len(data.image_files), selected_frames=len(files)))
            except Exception as exc:
                failures.append(dict(dataset=name, scene=scene, error=str(exc)))
    write_json(a.output, dict(scenes=len(records), expected_scenes=sum(SCENE_COUNTS.values()),
                             selected_images=sum(r['selected_frames'] for r in records), records=records, failures=failures))
    print(f'{len(records)}/90 scenes validated; {len(failures)} failures; {a.output}')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
