import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
from addict import Dict
from self_geometry import ROOT
from self_geometry.common import digest, write_json
from self_geometry.data import sampled_indices
from self_geometry.training import identity
from depth_anything_3.bench.evaluator import Evaluator


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_sampling_matches_da3_and_small_scenes_ignore_seed():
    raw = Dict(image_files=[str(i) for i in range(397)],
               extrinsics=np.zeros((397, 4, 4)), intrinsics=np.zeros((397, 3, 3)), aux=Dict())
    official = Evaluator._sample_frames(SimpleNamespace(max_frames=100), raw, 'test')
    assert sampled_indices(397, 100, 42) == [int(p) for p in official.image_files]
    assert sampled_indices(397, 100, 42) != sampled_indices(397, 100, 43)
    for count in (23, 49, 100):
        assert all(sampled_indices(count, 100, s) == list(range(count)) for s in (42, 43, 44))
    state = random.getstate()
    sampled_indices(397, 100, 44)
    assert random.getstate() == state


def test_same_frames_reuse_prediction_without_loading_model(tmp_path, monkeypatch):
    module = script('evaluate_seeds')
    source, output = tmp_path/'source', tmp_path/'evaluation'
    weights = tmp_path/'weights'; weights.write_bytes(b'weights')
    image = tmp_path/'rgb'; image.write_bytes(b'image')
    manifest = dict(dataset='eth3d', scene='test', image_files=[str(image)],
                    images=[dict(path=str(image), bytes=image.stat().st_size, mtime_ns=image.stat().st_mtime_ns)])
    manifest['fingerprint'] = digest(manifest)
    c = dict(method='baseline', seed=0, threads=1, weights=str(weights))
    write_json(source/'manifest.json', manifest)
    write_json(source/'baseline/protocol.json', dict(config=c, identity=digest(identity(c, manifest))))
    prediction = source/'baseline/exports/mini_npz/results.npz'
    prediction.parent.mkdir(parents=True); np.savez(prediction, depth=np.ones((1, 2, 2)))
    def prepare(name, scene, c, directory, sampling_seed):
        result = dict(manifest, sampling_seed=sampling_seed)
        result['fingerprint'] = digest(result)
        write_json(directory/'manifest.json', result)
        return result
    monkeypatch.setattr(module, 'prepare', prepare)
    monkeypatch.setattr(module, 'seed_all', lambda *a: None)
    def forbidden(*a, **kw):
        raise AssertionError('Identical frames must not load a GPU model')
    monkeypatch.setattr('self_geometry.model.load_model', forbidden)
    records = module.infer_samples(source, output, [42, 43, 44])
    assert {r['canonical'] for r in records} == {str(source)}
    assert len(records) == 3
    assert module.infer_samples(source, output, [42, 43, 44]) == records


def test_evaluation_averaging_counts_aliases_and_withholds_incomplete_mean(tmp_path):
    module = script('summarize_eval_seeds')
    from self_geometry.benchmark import metrics_for
    source, output = tmp_path/'source', tmp_path/'eval'
    write_json(source/'matrix_plan.json', dict(methods=['tco'], models=['da3'], scenes={'eth3d': ['one']}))
    write_json(output/'plan.json', dict(source=str(source), eval_seeds=[42, 43, 44]))
    canonical = source/'one'
    write_json(canonical/'adapted/metrics.json', {k: .5 for k in metrics_for('eth3d')})
    for seed in (42, 43, 44):
        write_json(output/f'tco/da3/eval_seed_{seed}/eth3d/one/evaluation_ref.json',
                   dict(canonical_directory=str(canonical), stage='adapted', checkpoint_sha256='same'))
    group = module.summarize(output)['groups'][0]
    assert group['complete'] and group['unique_evaluations'] == 1 and group['completed'] == 3
    assert group['metrics']['auc03'] == dict(mean=.5, sample_std=0.)
    (output/'tco/da3/eval_seed_44/eth3d/one/evaluation_ref.json').unlink()
    group = module.summarize(output)['groups'][0]
    assert not group['complete'] and not group['metrics']
