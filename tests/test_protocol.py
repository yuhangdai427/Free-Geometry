import pytest
from self_geometry import ROOT
from self_geometry.common import config
from self_geometry.protocol import paper_protocol


def test_formal_protocol_accepts_runtime_only_optimizations():
    c = config(ROOT/'configs/comparison.yaml')
    c.update(checkpointing=False, checkpoint_every=25, test3r_pair_batch=1)
    result = paper_protocol(c, ['baseline', 'self_geometry', 'tco', 'test3r'])
    assert result['profile'] == 'self_geometry_paper'
    assert result['checked_settings']['max_frames'] == 100


@pytest.mark.parametrize('overrides,skip', [
    ({'max_frames': 3}, False), ({'iterations': 2}, False),
    ({'test3r_max_updates': 50}, False), ({'tco_steps': 2}, False),
    ({}, True),
])
def test_formal_protocol_rejects_smoke_and_budget_overrides(overrides, skip):
    c = config(ROOT/'configs/comparison.yaml'); c.update(overrides)
    with pytest.raises(ValueError, match='Formal protocol mismatch'):
        paper_protocol(c, ['self_geometry', 'test3r', 'tco'], skip)
