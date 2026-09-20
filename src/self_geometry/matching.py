from pathlib import Path
import time
import torch
from .common import save_torch


def matches(images, c, path, fingerprint):
    path=Path(path)
    identity=dict(fingerprint=fingerprint,image_size=c['image_size'],keypoints=c['keypoints'],matcher='superpoint-lightglue',preprocessing=c.get('model','vggt'))
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        saved['identity'].setdefault('preprocessing', 'vggt')
        if saved['identity']!=identity: raise ValueError('Match cache identity mismatch')
        return saved['pairs']
    from lightglue import SuperPoint, LightGlue
    extractor=SuperPoint(max_num_keypoints=c['keypoints']).eval().to(images.device)
    matcher=LightGlue(features='superpoint').eval().to(images.device)
    start=time.monotonic();pairs={}
    with torch.inference_mode():
        features=[extractor.extract(im,resize=None) for im in images]
        for i in range(len(images)):
            for j in range(i+1,len(images)):
                out=matcher({'image0':features[i],'image1':features[j]})
                indices=out['matches'][0]
                pairs[(i,j)]=(features[i]['keypoints'][0,indices[:,0]].cpu(),features[j]['keypoints'][0,indices[:,1]].cpu())
            print(f'matching {i+1}/{len(images)} pairs={len(pairs)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    # Clone outside inference_mode so tensors can participate in backward.
    pairs={k:(a.clone(),b.clone()) for k,(a,b) in pairs.items()}
    save_torch(path,dict(identity=identity,pairs=pairs,seconds=time.monotonic()-start))
    del extractor,matcher,features
    torch.cuda.empty_cache()
    return pairs


def get_pair(pairs,i,j,device):
    a,b=pairs[(min(i,j),max(i,j))]
    if i>j: a,b=b,a
    return a.to(device),b.to(device)
