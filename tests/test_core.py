import math
import torch
from torch import nn
from self_geometry.geometry import project,residuals,filter_mask,sample,huber,threshold
from self_geometry.optimization import fan_setup,fan_sample,project_gradients,dwa,learning_rate,combine_gradients
from self_geometry.model import LoRALinear,inject_lora,trainable_state,load_trainable
from self_geometry.common import config


def scene():
    k=torch.tensor([[100.,0,20],[0,100,20],[0,0,1]])
    a=torch.eye(4)[:3];b=a.clone();b[0,3]=.1
    return k,a,b


def test_geometry_exact_and_depth_gradient():
    k,a,b=scene();xy=torch.tensor([[20.,20.],[21,22.]])
    depth=torch.full((2,),2.)
    uv,z=project(xy,depth,k,k,a,b)
    torch.testing.assert_close(uv,xy+torch.tensor([5.,0]))
    pred=dict(depth=torch.full((2,41,41),2.,requires_grad=True),intrinsics=torch.stack([k,k]),extrinsics=torch.stack([b,a]))
    ec,mvc,valid=residuals(pred,0,1,uv,xy)
    assert valid.all() and ec.max()<1e-5 and mvc.max()<1e-5
    ec2,mvc2,valid=residuals(pred,0,1,uv+torch.tensor([1.,1.]),xy)
    mvc2.mean().backward()
    assert pred['depth'].grad.abs().sum()>0
    assert ec2.min()>0


def test_camera_gradient_and_ec_independent_of_depth():
    k,a,b=scene();b=b.clone().requires_grad_()
    d=torch.full((2,41,41),2.,requires_grad=True)
    pred=dict(depth=d,intrinsics=torch.stack([k,k]),extrinsics=torch.stack([b,a]))
    xy=torch.tensor([[20.,20.],[21,22.]])
    ec,_,_=residuals(pred,0,1,xy+torch.tensor([4.,2.]),xy)
    g=torch.autograd.grad(ec.sum(),[b,d],allow_unused=True)
    assert g[0].abs().sum()>0 and g[1] is None


def test_sampling_pixels_and_filter_empty():
    im=torch.arange(25.).reshape(1,5,5)
    torch.testing.assert_close(sample(im,torch.tensor([[0.,0.],[4.,4.],[2.,1.]]))[:,0],torch.tensor([0.,24.,7.]))
    assert not filter_mask(torch.zeros(0),torch.zeros(0),torch.zeros(0,dtype=torch.bool),.9).any()
    r=torch.tensor([1.,2.,100.]);m=filter_mask(r,r,torch.ones(3,dtype=torch.bool),.9)
    assert not m[-1]


def test_projection_orthogonal_and_zero():
    a=[torch.tensor([1.,2.]),torch.tensor([3.])];b=[torch.tensor([3.,-1.]),torch.tensor([2.])]
    out,_,_=project_gradients(a,b)
    assert abs(sum((x*y).sum() for x,y in zip(out,b)))<1e-5
    out,_,_=project_gradients(a,[torch.zeros_like(x) for x in a])
    for x,y in zip(out,a):torch.testing.assert_close(x,y)


def test_combined_gradient():
    p=nn.Parameter(torch.tensor([1.,2.]))
    mvc=(p[0]+p[1]).square();ec=p[0].square();aux=p[1].square()
    combine_gradients([mvc,ec,aux,aux*0,aux*0],[p],[1]*5,'mvc_orthogonal_ec')
    torch.testing.assert_close(p.grad,torch.tensor([2.,10.]))


def test_lora_zero_init_and_frozen():
    linear=nn.Linear(3,4);layer=LoRALinear(linear,2,2,.1)
    x=torch.randn(5,3)
    torch.testing.assert_close(layer(x),linear(x))
    layer(x).sum().backward()
    assert linear.weight.grad is None and layer.b.grad.abs().sum()>0
    saved=trainable_state(layer)
    with torch.no_grad():layer.b.add_(1)
    load_trainable(layer,saved)
    torch.testing.assert_close(layer(x),linear(x))


def test_fan_and_dwa_schedule():
    angles=torch.tensor([0.,10.,40.,180.])*math.pi/180
    r=torch.stack([torch.tensor([[a.cos(),-a.sin(),0],[a.sin(),a.cos(),0],[0,0,1]]) for a in angles])
    target,groups=fan_setup(r)
    idx=fan_sample(target,groups,0)
    assert len(set(idx))==len(idx) and target not in sum(groups,[])
    assert sum(len(g) for g in groups)==3
    assert dwa([])==[1.]*5
    assert abs(sum(dwa([[1]*5,[100,0,2,1,1]]))-5)<1e-5
    c=config();assert learning_rate(0,c)<c['lr']
    assert abs(learning_rate(49,c)-c['final_lr'])<1e-12


def test_huber_zero_scale_finite():
    x=torch.zeros(10,requires_grad=True)
    value=huber(x,threshold(x)).mean();value.backward()
    assert torch.isfinite(x.grad).all()


def test_conflict_only_preserves_aligned_tasks():
    p=nn.Parameter(torch.tensor([1.,2.]))
    mvc=(p[0]+p[1]).square();ec=p[0].square();zero=p.sum()*0
    combine_gradients([mvc,ec,zero,zero,zero],[p],[1]*5,'mvc_conflict_only')
    torch.testing.assert_close(p.grad,torch.tensor([8.,6.]))
