import torch
from torch import nn
from self_geometry.comparisons import PromptBlock, triplet_order, tco_settings, remove_adapters
from self_geometry.comparisons import CachedEncoderBlock


class Mixer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d))
    def forward(self,x):
        return x @ self.weight + x.mean(dim=1,keepdim=True)


def test_deep_prompt_strip_frozen_parameters_and_reset():
    model = nn.Sequential(*[Mixer(5) for _ in range(4)])
    model.requires_grad_(False)
    x = torch.randn(2,7,5)
    frozen = model(x).detach().clone()
    for i in range(4):
        model[i] = PromptBlock(model[i],5,3,i,i==3)
    y = model(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert all(p.grad is None for n,p in model.named_parameters() if 'base' in n)
    assert all(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'prompt' in n)
    # remove_adapters supports actual nested architecture paths.
    outer = nn.Module(); outer.encoder = model
    remove_adapters(outer)
    torch.testing.assert_close(outer.encoder(x),frozen,rtol=0,atol=0)


def test_prompt_batched_gradient_matches_independent_pairs():
    torch.manual_seed(7)
    m = nn.Sequential(PromptBlock(Mixer(4),4,2,0,False),PromptBlock(Mixer(4),4,2,1,True))
    m[0].base.requires_grad_(False);m[1].base.requires_grad_(False)
    x = torch.randn(4,5,4)
    m(x).square().mean().backward()
    expected = m[0].prompt.grad.clone()
    m.zero_grad(set_to_none=True)
    for sample in x:
        (m(sample[None]).square().mean()/len(x)).backward()
    torch.testing.assert_close(m[0].prompt.grad,expected)


def test_test3r_exhaustive_population_and_seed():
    import random
    expected = list(range(4**3));random.Random(2).shuffle(expected)
    assert triplet_order(4,2) == expected
    assert sorted(triplet_order(4,2)) == list(range(64))
    assert triplet_order(4,2) != triplet_order(4,1)
    short = triplet_order(4,2,8)
    assert len(short)==len(set(short))==8


def test_tco_dataset_defaults_and_explicit_zero_photo():
    assert tco_settings({},'eth3d') == dict(steps=40,lr=5e-4,photo=.2)
    assert tco_settings({},'dtu') == dict(steps=50,lr=2e-4,photo=1.)
    assert tco_settings({'tco_steps':2,'tco_photo_weight':0},'hiroom')['photo']==0


def test_frozen_prefix_cache_survives_inplace_camera_token_replacement():
    class Counter(nn.Module):
        def __init__(self):
            super().__init__(); self.calls=0
        def forward(self,x):
            self.calls+=1
            return x+2
    base=[Counter(),Counter()]
    cache={}
    model=nn.Sequential(CachedEncoderBlock(base[0],cache),
                        CachedEncoderBlock(base[1],cache,last=True,clone=True))
    x=torch.zeros(2,5,3)
    first=model(x)
    first[:,0]=100
    second=model(x)
    torch.testing.assert_close(second,torch.full_like(x,4))
    assert [b.calls for b in base]==[1,1]
