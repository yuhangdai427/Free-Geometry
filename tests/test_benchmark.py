import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from addict import Dict
from self_geometry import ROOT
from self_geometry.aggregate import aggregate_seeds
from self_geometry.benchmark import DATASETS, SCENE_COUNTS, metrics_for
from self_geometry.common import config, write_json
from self_geometry.data import evaluation_data
from self_geometry.report import collect


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_report_distinguishes_true_zero_from_missing_evaluation(tmp_path):
    report = script('summarize_comparison')
    jobs = [dict(model='vggt', method='baseline', dataset='eth3d', scene=s, seed=0)
            for s in ('evaluated', 'pending')]
    directory = tmp_path/'baseline/vggt/seed_0/eth3d/evaluated'
    write_json(directory/'manifest.json', {'image_files': ['a', 'b']})
    write_json(directory/'baseline/metrics.json', {key: 0. for key in metrics_for('eth3d')})
    result = report.selected_results(tmp_path, {'jobs': jobs})
    assert result['complete'] == 1 and result['total'] == 2
    assert result['rows'][0]['status'] == 'evaluated'
    assert result['rows'][0]['metrics']['recon_unposed_fscore'] == 0.
    assert result['rows'][1]['status'] == 'pending' and not result['rows'][1]['metrics']
    assert 'not dataset averages' in result['scope']


def test_full_plan_contains_all_270_unique_jobs(monkeypatch):
    runner = script('run_seeds_separately')
    monkeypatch.setattr(runner, 'dataset', lambda name, c: SimpleNamespace(SCENES=[f's{i}' for i in range(SCENE_COUNTS[name])]))
    plan = runner.build_plan(config(), list(DATASETS), [0, 1, 2])
    assert plan['adaptations'] == 270
    assert len({(j['seed'], j['dataset'], j['scene']) for j in plan['jobs']}) == 270
    assert sum(j['dataset'] == 'dtu' for j in plan['jobs']) == 66
    assert plan['sampling_seed'] == 42


def test_dtu_selection_preserves_mask_camera_and_rgb_order():
    data = Dict(image_files=['ref33', 'frame0', 'frame1'], extrinsics=np.arange(48).reshape(3, 4, 4),
                intrinsics=np.arange(27).reshape(3, 3, 3), aux=Dict(mask_files=['mask33', 'mask0', 'mask1']))
    ds = SimpleNamespace(get_data=lambda _: data)
    selected = evaluation_data(ds, 'scan1', {'image_files': ['frame1', 'ref33']})
    assert selected.aux.mask_files == ['mask1', 'mask33']
    np.testing.assert_array_equal(selected.extrinsics, data.extrinsics[[2, 0]])
    np.testing.assert_array_equal(selected.intrinsics, data.intrinsics[[2, 0]])
    with pytest.raises(ValueError, match='Duplicate'):
        evaluation_data(ds, 'scan1', {'image_files': ['ref33', 'ref33']})


def test_seed_stats_require_full_coverage_and_keep_regressions():
    summaries = {}
    for seed in [0, 1, 2]:
        summaries[str(seed)] = {'datasets': {'dtu': {
            stage: dict(complete=True, completed=22, expected=22,
                        means={k: float(seed + offset) for k in metrics_for('dtu')})
            for stage, offset in [('baseline', 1), ('adapted', 3)]}}}
    out = aggregate_seeds(summaries, ['dtu'], [0, 1, 2])['dtu']
    stat = out['adapted']['metrics']['recon_unposed_overall']
    assert stat == dict(mean=4., sample_std=1., direction='lower')
    assert out['paired_adapted_minus_baseline']['recon_unposed_overall']['mean'] == 2.
    assert 'recon_unposed_fscore' not in out['adapted']['metrics']
    summaries['1']['datasets']['dtu']['adapted'].update(complete=False, completed=21)
    out = aggregate_seeds(summaries, ['dtu'], [0, 1, 2])['dtu']
    assert not out['adapted']['complete'] and out['adapted']['metrics'] == {}
    assert out['paired_adapted_minus_baseline'] == {}


def test_report_is_scene_macro_and_missing_dtu_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr('self_geometry.data.dataset', lambda name, c: SimpleNamespace(SCENES=['a', 'b']))
    for scene, value in [('a', 1.), ('b', 3.)]:
        write_json(tmp_path / 'dtu' / scene / 'baseline/metrics.json', {k: value for k in metrics_for('dtu')})
    write_json(tmp_path / 'dtu/a/adapted/metrics.json', {k: 20. for k in metrics_for('dtu')})
    result, rows = collect(tmp_path, {}, ['dtu'])
    assert result['datasets']['dtu']['baseline']['means']['recon_posed_overall'] == 2.
    assert result['datasets']['dtu']['adapted']['completed'] == 1
    assert not result['datasets']['dtu']['adapted']['complete']
    assert len(rows) == 3 and len(result['missing']) == 1


def test_campaign_continues_after_scene_failure_and_serializes_dtu(tmp_path, monkeypatch):
    runner = script('campaign')
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(runner, 'dataset', lambda name, c: SimpleNamespace(SCENES=['bad', 'good']))
    monkeypatch.setattr(runner, 'report', lambda *args: None)
    calls = []
    def run(cmd, **kwargs):
        stage = cmd[2]
        scene = cmd[cmd.index('--scene') + 1]
        calls.append((stage, scene))
        return SimpleNamespace(returncode=1 if stage == 'baseline' and scene == 'bad' else 0)
    monkeypatch.setattr(runner.subprocess, 'run', run)
    ret = runner.main(['--output', str(tmp_path), '--datasets', 'dtu'])
    assert ret == 1
    assert calls == [('baseline', 'bad'), ('baseline', 'good'), ('evaluate', 'good'),
                     ('adapt', 'good'), ('evaluate', 'good')]
    status = json.loads((tmp_path / 'campaign.json').read_text())
    assert status['active'] == [] and 'finished' in status


def test_full_runner_continues_after_failed_seed(tmp_path, monkeypatch):
    runner = script('run_seeds_separately')
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(runner, 'dataset', lambda name, c: SimpleNamespace(SCENES=['scan1']))
    seen = []
    def run(cmd, **kwargs):
        if cmd[1].endswith('campaign.py'):
            c = json.loads(Path(cmd[cmd.index('--config') + 1]).read_text())
            seen.append(c['seed'])
            return SimpleNamespace(returncode=int(c['seed'] == 0))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', run)
    assert runner.main(['--root', str(tmp_path), '--datasets', 'dtu', '--no-reuse']) == 1
    assert seen == [0, 1, 2]
    status = json.loads((tmp_path / 'experiments.json').read_text())
    assert status['status'] == 'finished_with_failures'


def test_cache_build_does_not_advance_training_rng():
    import random
    import torch
    from self_geometry.common import seed_all, isolated_rng
    seed_all(7)
    with isolated_rng():
        random.random()
        np.random.rand(21)
        torch.rand(100)
    actual = (random.random(), np.random.rand(), torch.rand(5))
    seed_all(7)
    expected = (random.random(), np.random.rand(), torch.rand(5))
    assert actual[0:2] == expected[0:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_cache_reuses_baseline_but_never_other_seed_adaptation(tmp_path):
    from self_geometry.cache import reuse
    from self_geometry.common import digest
    from self_geometry.training import identity
    weights = tmp_path / 'model.pt'
    weights.write_bytes(b'weights')
    rgb = tmp_path / 'rgb.png'
    rgb.write_bytes(b'rgb')
    c = dict(config(), weights=str(weights))
    manifest = dict(dataset='dtu', scene='scan1', image_files=[str(rgb)], fingerprint='test',
                    images=[dict(path=str(rgb), bytes=rgb.stat().st_size, mtime_ns=rgb.stat().st_mtime_ns)])
    scene = tmp_path / 'source/dtu/scan1'
    write_json(scene / 'manifest.json', manifest)
    write_json(scene / 'baseline/protocol.json', dict(config=c, identity=digest(identity(c, manifest))))
    pred = scene / 'baseline/exports/mini_npz/results.npz'
    pred.parent.mkdir(parents=True)
    np.savez(pred, depth=np.ones((1, 1, 1)))
    write_json(scene / 'adapted/complete.json', dict(config=c, identity=digest(identity(c, manifest))))
    # A degraded metric is still reused at the same seed/config, not discarded.
    write_json(scene / 'adapted/metrics.json', {'auc03': 0.006})
    assert reuse(tmp_path / 'source', tmp_path / 'same', c, completed=True) == 1
    assert (tmp_path / 'same/dtu/scan1/adapted/metrics.json').exists()
    assert reuse(tmp_path / 'source', tmp_path / 'other', dict(c, seed=1), completed=True) == 1
    assert (tmp_path / 'other/dtu/scan1/baseline/protocol.json').exists()
    assert not (tmp_path / 'other/dtu/scan1/adapted/complete.json').exists()


def test_dual_plan_540_adaptations_180_shared_workers(monkeypatch):
    runner = script('run_full')
    monkeypatch.setattr(runner, 'dataset', lambda name, c: SimpleNamespace(SCENES=[f's{i}' for i in range(SCENE_COUNTS[name])]))
    configs = {m: config(model=m) for m in ['da3', 'vggt']}
    plan = runner.build_plan(configs, list(DATASETS), [0, 1, 2])
    assert plan['adaptations'] == 540 and plan['scene_workers'] == 180
    assert len({(x['model'], x['dataset'], x['scene'], x['seed']) for x in plan['jobs']}) == 540
    assert configs['da3']['weights'].endswith('model.safetensors')
    assert configs['vggt']['weights'].endswith('model.pt')


def test_lora_reset_preserves_base_and_changes_seed_initialization():
    import torch
    from torch import nn
    from self_geometry.model import inject_lora, remove_lora
    from self_geometry.common import seed_all
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.pretrained = nn.Module()
    model.backbone.pretrained.attn = nn.Module()
    model.backbone.pretrained.attn.qkv = nn.Linear(8, 24)
    original = model.backbone.pretrained.attn.qkv
    c = config(model='da3')
    seed_all(0)
    inject_lora(model, c)
    first = model.backbone.pretrained.attn.qkv.a.detach().clone()
    with torch.no_grad():
        model.backbone.pretrained.attn.qkv.b.fill_(1)
    remove_lora(model)
    assert model.backbone.pretrained.attn.qkv is original
    seed_all(1)
    inject_lora(model, c)
    assert not torch.equal(first, model.backbone.pretrained.attn.qkv.a)
    assert torch.count_nonzero(model.backbone.pretrained.attn.qkv.b) == 0
    remove_lora(model)
    seed_all(0)
    inject_lora(model, c)
    torch.testing.assert_close(first, model.backbone.pretrained.attn.qkv.a, rtol=0, atol=0)


def test_da3_preprocessing_matches_official_normalized_input(tmp_path):
    from PIL import Image
    import torch
    from self_geometry.model import load_images
    from depth_anything_3.utils.io.input_processor import InputProcessor
    image = np.random.default_rng(3).integers(0, 256, (81, 105, 3), dtype=np.uint8)
    path = tmp_path / 'rgb.png'
    Image.fromarray(image).save(path)
    raw = load_images([str(path)], 56, 'da3')
    official, _, _ = InputProcessor()([str(path)], process_res=56, sequential=True)
    mean = raw.new_tensor([.485, .456, .406])[None, :, None, None]
    std = raw.new_tensor([.229, .224, .225])[None, :, None, None]
    torch.testing.assert_close((raw - mean) / std, official, rtol=0, atol=0)
    assert float(raw.min()) >= 0 and float(raw.max()) <= 1


def test_json_config_roundtrip_scientific_notation(tmp_path):
    c = config(model='da3')
    path = tmp_path / 'config.json'
    write_json(path, c)
    assert config(path) == c
    assert isinstance(config(path)['lr'], float)
    assert config(overrides=['lr=5e-5'])['lr'] == 0.00005


@pytest.mark.parametrize('method', ['self_geometry', 'test3r', 'tco'])
def test_shared_worker_continues_and_loads_model_once(tmp_path, monkeypatch, method):
    import torch
    worker = script('scene_worker')
    cfg = tmp_path / 'config.json'
    write_json(cfg, dict(config(model='da3'), method=method))
    loaded, seen = [], []
    monkeypatch.setattr(worker, 'load_model', lambda c: loaded.append(c['model']) or object())
    monkeypatch.setattr(worker, 'remove_adapters', lambda model: None)
    monkeypatch.setattr(worker, 'load_images', lambda *a: torch.zeros(2, 3, 14, 14))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(worker, 'reuse_baseline', lambda *a: False)
    def prepare(name, scene, c, directory):
        write_json(directory / 'manifest.json', {'image_files': ['a', 'b']})
    def baseline(c, directory, **kwargs):
        write_json(directory / 'baseline/protocol.json', {})
    def adapt(c, directory, **kwargs):
        seen.append(c['seed'])
        if c['seed'] == 0:
            raise RuntimeError('simulated seed failure')
    monkeypatch.setattr(worker, 'prepare', prepare)
    monkeypatch.setattr(worker, 'baseline', baseline)
    monkeypatch.setattr(worker, 'adapt', adapt)
    monkeypatch.setattr(worker, 'adapt_comparison', lambda c, d, name, **kw: adapt(c, d, **kw))
    assert worker.main(['--config', str(cfg), '--root', str(tmp_path / 'runs'), '--dataset', 'dtu',
                        '--scene', 'scan1', '--seeds', '0', '1', '2']) == 1
    assert loaded == ['da3'] and seen == [0, 1, 2]
    result = json.loads((tmp_path / 'runs/workers/dtu/scan1/result.json').read_text())
    assert [r['exit_code'] for r in result['runs']] == [1, 0, 0]


def test_dtu_split_fusion_and_scoring_match_combined(tmp_path, monkeypatch):
    from self_geometry import data
    cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    cameras[1, 0, 3] = 1
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    calls = []
    class FakeDTU:
        def get_data(self, scene):
            return Dict(image_files=['a', 'b'], extrinsics=cameras, intrinsics=intrinsics, aux=Dict(mask_files=['ma', 'mb']))
        def fuse3d(self, scene, result, path, mode):
            calls.append(('fuse', mode))
            Path(path).write_text('pointcloud')
        def eval3d(self, scene, path):
            calls.append(('score', Path(path).parent.name))
            return dict(acc=1., comp=2., overall=1.5)
    monkeypatch.setattr(data, 'dataset', lambda *args: FakeDTU())
    def make(name):
        directory = tmp_path / name
        write_json(directory / 'manifest.json', dict(image_files=['a', 'b'], fingerprint='test'))
        np.savez(directory / 'gt_meta.npz', image_files=['a', 'b'], extrinsics=cameras, intrinsics=intrinsics)
        export = directory / 'baseline/exports/mini_npz'
        export.mkdir(parents=True)
        np.savez(export / 'results.npz', depth=np.ones((2, 2, 2)), extrinsics=cameras[:, :3], intrinsics=intrinsics)
        return directory
    together, split = make('together'), make('split')
    c = config(model='da3')
    with pytest.raises(ValueError, match='Run fuse'):
        data.evaluate('dtu', 'scan1', c, split, 'baseline', score_only=True)
    expected = data.evaluate('dtu', 'scan1', c, together, 'baseline')
    data.evaluate('dtu', 'scan1', c, split, 'baseline', fuse_only=True)
    assert not (split / 'baseline/metrics.json').exists()
    actual = data.evaluate('dtu', 'scan1', c, split, 'baseline', score_only=True)
    assert actual == expected
    assert sum(k == 'fuse' for k, _ in calls) == 4
    assert sum(k == 'score' for k, _ in calls) == 4


def test_suite_dtu_fuses_on_gpu_scores_on_cpu_and_shares_baseline(tmp_path, monkeypatch):
    runner = script('run_full')
    monkeypatch.setattr(runner, 'dataset', lambda *args: SimpleNamespace(SCENES=['scan1']))
    calls = []
    def run(cmd, **kwargs):
        if cmd[1].endswith('scene_worker.py'):
            for seed in [0, 1]:
                folder = tmp_path / 'da3' / f'seed_{seed}' / 'dtu/scan1'
                write_json(folder / 'baseline/protocol.json', {})
                for stage in ['baseline', 'adapted']:
                    path = folder / stage / 'exports/mini_npz/results.npz'
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b'prediction')
        elif cmd[1].endswith('run.py'):
            action, stage = cmd[2], cmd[cmd.index('--stage') + 1]
            calls.append((action, stage))
            if action == 'score':
                assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
                folder = Path(cmd[cmd.index('--output') + 1]) / 'dtu/scan1'
                write_json(folder / stage / 'metrics.json', {k: 1. for k in metrics_for('dtu')})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', run)
    assert runner.main(['--root', str(tmp_path), '--models', 'da3', '--datasets', 'dtu', '--seeds', '0', '1']) == 0
    assert calls.count(('fuse', 'baseline')) == calls.count(('score', 'baseline')) == 1
    assert calls.count(('fuse', 'adapted')) == calls.count(('score', 'adapted')) == 2
    assert (tmp_path / 'da3/seed_1/dtu/scan1/baseline/metrics.json').exists()
