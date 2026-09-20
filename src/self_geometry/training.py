import json
from contextlib import contextmanager
import fcntl
import math
import time
from pathlib import Path
import numpy as np
import torch
from .common import digest,write_json,save_torch,seed_all,rng_state,restore_rng,isolated_rng
from .model import load_images,load_model,predict,inject_lora,trainable_state,load_trainable,training_mode,remove_lora
from .matching import matches,get_pair
from .geometry import residuals,filter_mask,threshold,huber,photo_residual,smoothness
from .optimization import fan_setup,fan_sample,dwa,learning_rate,combine_gradients


def prediction_path(directory,stage): return Path(directory)/stage/'exports/mini_npz/results.npz'


@contextmanager
def stage_lock(directory,stage):
    path=Path(directory)/stage
    path.mkdir(parents=True,exist_ok=True)
    with (path/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def export(pred,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    values={k:v.detach().cpu().numpy() for k,v in pred.items()}
    if not all(np.isfinite(v).all() for v in values.values()): raise ValueError('Nonfinite prediction')
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**values);tmp.replace(path)


def load_prediction(path,device):
    with np.load(path) as z: return {k:torch.from_numpy(z[k]).to(device) for k in z.files}


def identity(c,manifest):
    p=Path(c['weights'])
    return dict(config=c,manifest=manifest['fingerprint'],weight_file=str(p),weight_size=p.stat().st_size,
                weight_mtime=p.stat().st_mtime_ns)


def baseline(c, directory, model=None, images=None):
    with stage_lock(directory,'baseline'):
        return _baseline(c,directory,model,images)


def _baseline(c, directory, model=None, images=None):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    sig=digest(identity(c,manifest));path=prediction_path(directory,'baseline')
    stamp=directory/'baseline/protocol.json'
    if path.exists() and stamp.exists():
        if json.loads(stamp.read_text())['identity']!=sig: raise ValueError('Baseline identity mismatch')
        return
    seed_all(c['seed'],c['threads']);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();model=load_model(c) if model is None else model
    if images is None: images=load_images(manifest['image_files'],c['image_size'],c.get('model','vggt')).cuda()
    with torch.no_grad(): pred=predict(model,images,c)
    export(pred,path)
    write_json(stamp,dict(identity=sig,config=c,frames=len(images),shape=list(images.shape),
                         seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/1024**3))


def initialize_losses(base,images,pairs,target,c):
    ec_all=[];mvc_all=[];masks={}
    # Thresholds use baseline residuals across all pairs, once per scene.
    with torch.no_grad():
        for (i,j) in pairs:
            xi,xj=get_pair(pairs,i,j,images.device)
            if not len(xi): continue
            ec,mvc,valid=residuals(base,i,j,xi,xj)
            mask=filter_mask(ec,mvc,valid,c['filter_keep'])
            masks[(i,j)]=mask.cpu()
            if mask.any(): ec_all.append(ec[mask]);mvc_all.append(mvc[mask])
        if not ec_all: raise RuntimeError('No valid baseline correspondences in scene')
        # Additional top-10% removal is for robust scale only, per appendix S.III.
        def trimmed_scale(values):
            r=torch.cat(values);r=r[r<=torch.quantile(r,0.9)]
            return threshold(r)
        delta=dict(ec=trimmed_scale(ec_all),mvc=trimmed_scale(mvc_all))
        photos=[photo_residual(images,base,target,j) for j in range(len(images)) if j!=target]
        photos=[p for p in photos if p.numel()]
        delta['pc']=threshold(torch.cat(photos)) if photos else 1e-6
    return delta,masks


def losses_for_subset(pred,images,indices,base,pairs,deltas,initial_masks,c):
    ecs=[];mvcs=[];pcs=[];count=0
    target=indices[0]
    for local_j,global_j in enumerate(indices[1:],1):
        xi,xj=get_pair(pairs,target,global_j,images.device)
        if not len(xi): continue
        ec,mvc,valid=residuals(pred,0,local_j,xi,xj)
        if c['filter_each_step']:
            mask=filter_mask(ec,mvc,valid,c['filter_keep'])
        else:
            # Masks were computed in ascending pair orientation. Recompute initial
            # direction for this target so MVC uses the correct source depth.
            bec,bmvc,bvalid=residuals(base,target,global_j,xi,xj)
            mask=filter_mask(bec,bmvc,bvalid,c['filter_keep'])&valid.detach()
        if mask.any():
            ecs.append(ec[mask]);mvcs.append(mvc[mask]);count+=int(mask.sum())
        p=photo_residual(images,pred,0,local_j)
        if p.numel(): pcs.append(p)
    if not count: return None,0
    mvc=huber(torch.cat(mvcs),deltas['mvc']).mean()
    ec=huber(torch.cat(ecs),deltas['ec']).mean()
    zero=pred['depth'].sum()*0
    pc=huber(torch.cat(pcs),deltas['pc']).mean() if pcs else zero
    eds=smoothness(pred['depth'][0],images[0])
    confidence=base['conf'][target]
    confident=confidence>=confidence.quantile(0.5)
    anchor=(pred['depth'][0]-base['depth'][target]).abs()[confident]
    bdc=huber(anchor,threshold(anchor)).mean() if anchor.numel() else zero
    return [mvc,ec,pc,eds,bdc],count


def adapt(c,directory,resume=False,model=None,scene_cache=None):
    with stage_lock(directory,'adapted'):
        return _adapt(c,directory,resume,model,scene_cache)


def _adapt(c,directory,resume=False,model=None,scene_cache=None):
    directory=Path(directory);out=directory/'adapted';out.mkdir(exist_ok=True)
    manifest=json.loads((directory/'manifest.json').read_text());sig=digest(identity(c,manifest))
    complete=out/'complete.json'
    if complete.exists():
        if json.loads(complete.read_text())['identity']!=sig: raise ValueError('Adaptation identity mismatch')
        if prediction_path(directory,'adapted').exists(): return
    seed_all(c['seed'],c['threads']);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    scene_cache = {} if scene_cache is None else scene_cache
    images = scene_cache.get('images')
    if images is None: images=load_images(manifest['image_files'],c['image_size'],c.get('model','vggt')).cuda()
    scene_cache['images'] = images
    base = scene_cache.get('base')
    if base is None: base=load_prediction(prediction_path(directory,'baseline'),'cuda')
    scene_cache['base'] = base
    if json.loads((directory/'baseline/protocol.json').read_text())['identity']!=sig:
        raise ValueError('Baseline/config mismatch')
    if c['rng_protocol'] != 'isolated_preparation_v2':
        raise ValueError('Unsupported adaptation RNG protocol')
    with isolated_rng():
        pairs = scene_cache.get('pairs')
        if pairs is None: pairs=matches(images,c,directory/'matches.pt',manifest['fingerprint'])
        scene_cache['pairs'] = pairs
    model=load_model(c) if model is None else model
    remove_lora(model)
    names=inject_lora(model,c)
    params=[p for p in model.parameters() if p.requires_grad]
    write_json(out/'lora_modules.json',dict(modules=names,parameters=sum(p.numel() for p in params)))
    target,groups=fan_setup(base['extrinsics'][:,:3,:3],c['bin_width'],c['bins_compat9'])
    if not c['fan']:
        # Ablation keeps the FAN batch size, uses fixed first target and random sources.
        target=0
    initialization = scene_cache.get('initialization')
    if initialization is None: initialization=initialize_losses(base,images,pairs,target,c)
    scene_cache['initialization'] = initialization
    deltas,initial_masks = initialization
    write_json(out/'initialization.json',dict(target=target,groups=groups,thresholds=deltas,matching_pairs=len(pairs)))
    optimizer=torch.optim.AdamW(params,lr=c['lr'],weight_decay=c['weight_decay'],fused=c.get('fused_optimizer',False))
    history=[];best=float('inf');best_state=None;start=0;updates=0
    last=out/'last.pt'
    if last.exists():
        if not resume: raise FileExistsError('Partial adaptation exists; use --resume')
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['identity']!=sig: raise ValueError('Resume identity mismatch')
        load_trainable(model,state['parameters']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];start=state['next_iteration'];updates=state['updates']
        best_state=state.get('best_checkpoint')
        if best_state is None and (out/'best.pt').exists():
            best_state=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            if best_state['score']!=best: raise ValueError('Historical best does not match resume checkpoint')
        log_path=out/'steps.jsonl'
        if log_path.exists():
            lines=[line for line in log_path.read_text().splitlines() if json.loads(line)['iteration']<=start]
            log_path.write_text('\n'.join(lines)+'\n')
        restore_rng(state['rng'])
    elif (out/'steps.jsonl').exists(): raise FileExistsError('Uncheckpointed run; use a fresh directory')
    training_mode(model,c)
    for step in range(start,c['iterations']):
        tick=time.monotonic()
        if c['fan']: indices=fan_sample(target,groups,step)
        else:
            indices=[target]+np.random.choice([j for j in range(len(images)) if j!=target],size=len(groups),replace=False).tolist()
        batch=images[indices]
        optimizer.zero_grad(set_to_none=True)
        pred=predict(model,batch,c)
        losses,nmatches=losses_for_subset(pred,batch,indices,base,pairs,deltas,initial_masks,c)
        log=dict(iteration=step+1,indices=indices,matches=nmatches)
        if losses is None:
            log['skipped']='no_valid_geometry'
        else:
            numbers=[float(l.detach()) for l in losses]
            if not all(math.isfinite(x) for x in numbers): raise FloatingPointError(f'Losses: {numbers}')
            weights=dwa(history);score=math.sqrt(numbers[0]*numbers[1])
            # Loss was measured before optimizer.step: checkpoint those exact parameters.
            if score<best:
                best=score
                best_state=dict(identity=sig,parameters=trainable_state(model),score=score,
                                evaluated_iteration=step+1,updates=updates)
            cosine=combine_gradients(losses,params,weights,c['gd'])
            norm=torch.nn.utils.clip_grad_norm_(params,c['clip'],error_if_nonfinite=True)
            rate=learning_rate(step,c)
            for group in optimizer.param_groups: group['lr']=rate
            optimizer.step();updates+=1;history.append(numbers)
            log.update(losses=dict(zip(['mvc','ec','pc','eds','bdc'],numbers)),dwa=weights,score=score,
                       best=best,gradient_cosine=cosine,gradient_norm=float(norm),lr=rate,updates=updates)
        log['seconds']=time.monotonic()-tick
        with (out/'steps.jsonl').open('a') as f: f.write(json.dumps(log,allow_nan=False)+'\n')
        print(json.dumps(log),flush=True)
        del pred,losses,batch
        if (step+1)%c.get('checkpoint_every',5)==0 or step+1==c['iterations']:
            save_torch(last,dict(identity=sig,parameters=trainable_state(model),optimizer=optimizer.state_dict(),
                                history=history,best=best,best_checkpoint=best_state,
                                next_iteration=step+1,updates=updates,rng=rng_state()))
    if best_state is None: raise RuntimeError('No valid checkpoint')
    save_torch(out/'best.pt',best_state)
    load_trainable(model,best_state['parameters']);model.eval();optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    with torch.no_grad(): final=predict(model,images,c)
    export(final,prediction_path(directory,'adapted'))
    write_json(complete,dict(identity=sig,config=c,updates=updates,selected_updates=best_state['updates'],
                            selected_iteration=best_state['evaluated_iteration'],score=best,
                            seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/1024**3))
