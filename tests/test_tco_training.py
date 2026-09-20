import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from self_geometry import data
from self_geometry.comparisons import CachedEncoderBlock, uncache_tco_encoder
from self_geometry.model import checkpoint_depth_chunks


@pytest.mark.parametrize('name,count,expected', [
    ('eth3d', 38, list(range(0, 38, 5))),
    ('dtu', 49, list(range(0, 49, 5))),
    ('7scenes', 1000, [0, 200, 400, 600, 800]),
    ('7scenes', 500, [0, 200, 400]),
])
def test_original_strides(name, count, expected):
    indices, policy = data.tco_training_indices(name, count)
    assert indices == expected
    assert policy['strategy'] == 'official_stride'


def test_extensions_are_labeled_and_do_not_duplicate_small_scenes():
    for name in ('scannetpp', 'hiroom'):
        indices, policy = data.tco_training_indices(name, 397)
        assert len(indices) == len(set(indices)) == 10
        assert indices[0] == 0 and indices[-1] == 396
        assert policy['strategy'] == 'extension_uniform_10'
        assert data.tco_training_indices(name, 3)[0] == [0, 1, 2]


def test_training_sampling_independent_of_evaluation_cap(tmp_path, monkeypatch):
    files = []
    for i in range(38):
        p = tmp_path/f'{i}.png'; p.write_bytes(b'rgb'); files.append(str(p))
    ds = SimpleNamespace(get_data=lambda scene: SimpleNamespace(image_files=files))
    monkeypatch.setattr(data, 'dataset', lambda *args: ds)
    a = data.prepare_tco_training('eth3d', 'courtyard', {'max_frames': 100}, tmp_path/'train')
    b = data.prepare_tco_training('eth3d', 'courtyard', {'max_frames': 3}, tmp_path/'train')
    assert a == b
    assert a['image_files'] == files[::5]
    assert not (tmp_path/'train/gt_meta.npz').exists()
    (tmp_path/'0.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        data.prepare_tco_training('eth3d', 'courtyard', {}, tmp_path/'train')


def test_encoder_cache_must_be_removed_before_new_frame_inference():
    model = nn.Module()
    model.encoder = CachedEncoderBlock(nn.Identity(), {}, last=True)
    train = torch.randn(5, 3)
    evaluation = torch.randn(100, 3)
    model.encoder(train)
    assert model.encoder(evaluation).shape[0] == 5
    uncache_tco_encoder(model)
    torch.testing.assert_close(model.encoder(evaluation), evaluation)


def test_head_checkpoint_preserves_nested_output_and_input_gradient():
    class Head:
        def _forward_impl(self, feats, scale=1.):
            return {'depth': (feats[0].sin()*feats[1]).square()*scale}
    head = Head()
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4, requires_grad=True)
    expected = head._forward_impl([x, y], scale=2)['depth']
    gradients = torch.autograd.grad(expected.sum(), (x, y))
    checkpoint_depth_chunks(SimpleNamespace(_sg_checkpointing=True), head)
    actual = head._forward_impl([x, y], scale=2)['depth']
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad(actual.sum(), (x, y)), gradients):
        torch.testing.assert_close(got, want)
