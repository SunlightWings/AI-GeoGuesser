"""Isolated teacher CLI. Never imported by training or submission inference."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from common import cache_file, digest, json_write, metrics, norm, npz_write, read_embeddings, VERSION

def plan_load(path): return torch.load(Path(path)/'plan.pt',map_location='cpu',weights_only=False)

def teacher_load(name,device):
    if name=='geoclip':
        from geoclip import GeoCLIP
        model=GeoCLIP()
        processor=model.image_encoder.image_processor
        forward=lambda pixels:model.image_encoder(pixels)
    else:
        from transformers import CLIPModel, AutoProcessor
        model=CLIPModel.from_pretrained('geolocal/StreetCLIP')
        processor=AutoProcessor.from_pretrained('geolocal/StreetCLIP')
        forward=lambda pixels:model.get_image_features(pixel_values=pixels)
    # Fingerprint actual downloaded weights, not a moving repository branch.
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        if key in ('gps_queue','gps_queue_ptr','gps_gallery'): continue
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    signature=dict(version=VERSION,teacher=name,weights_sha256=h.hexdigest(),transformers=importlib.metadata.version('transformers'),processor=str(processor))
    model.eval().requires_grad_(False).to(device)
    return model,processor,forward,signature

@torch.inference_mode()
def cache(a):
    plan=plan_load(a.plan); ph=digest(Path(a.plan)/'plan.pt')
    tr=pd.read_csv(Path(a.plan)/'train_rows.csv'); va=pd.read_csv(Path(a.plan)/'validation_rows.csv')
    if not len(va) and not a.all: raise ValueError('Full-data plan requires --all')
    if a.all:
        rows=pd.concat([tr,va]).sort_values('filename')
    else:
        # Fixed, country-stratified-ish subset: shuffled round-robin across countries.
        groups=[g.sample(frac=1,random_state=2026).reset_index(drop=True) for _,g in va.groupby('country_id')]
        rows=pd.concat([g.assign(_rank=np.arange(len(g))) for g in groups]).sort_values(['_rank','country_id']).head(a.limit)
        if a.teacher=='streetclip': rows=pd.concat([rows,tr.sample(n=min(1024,len(tr)),random_state=2026)])
    device=torch.device(a.device)
    model,processor,forward,signature=teacher_load(a.teacher,device)
    root=Path(a.cache); root.mkdir(parents=True,exist_ok=True)
    meta=root/'teacher.json'
    if meta.exists() and json.loads(meta.read_text())!=signature: raise ValueError('Teacher cache version differs; use a new cache directory')
    json_write(meta,signature)
    image_root=Path(a.data)/'train'; pending=[]
    for name in rows.filename:
        path=image_root/name
        if not path.is_file(): raise FileNotFoundError(path)
        dst=cache_file(root,name)
        if dst.exists():
            with np.load(dst,allow_pickle=False) as z:
                if str(z['image_sha256'].item())!=digest(path): raise ValueError('Image changed since teacher cache creation')
        else: pending.append(name)
    print(f'{len(rows)-len(pending)} cached; {len(pending)} images to encode',flush=True)
    for start in range(0,len(pending),a.batch):
        names=pending[start:start+a.batch]; images=[]
        for name in names:
            with Image.open(image_root/name) as im: images.append(ImageOps.exif_transpose(im).convert('RGB'))
        pixels=processor(images=images,return_tensors='pt')['pixel_values'].to(device)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): emb=forward(pixels)
        if not torch.is_tensor(emb): raise TypeError('Unexpected teacher API; use the pinned transformers version')
        emb=norm(emb.float().cpu().numpy())
        for name,x in zip(names,emb): npz_write(cache_file(root,name),filename=np.asarray(name),embedding=x.astype(np.float16),image_sha256=np.asarray(digest(image_root/name)))
        print(f'Encoded {min(start+a.batch,len(pending))}/{len(pending)}',flush=True)
    candidates=Path(a.plan)/'geoclip_candidates.npz'
    if a.teacher=='geoclip' and not candidates.exists():
        gps=plan['fine']['output_centers_latlng']; encoded=[]
        for start in range(0,len(gps),256):
            encoded.append(model.location_encoder(torch.tensor(gps[start:start+256],dtype=torch.float32,device=device)).float().cpu().numpy())
        npz_write(candidates,latlng=gps,embedding=norm(np.concatenate(encoded)),logit_scale=np.asarray(float(model.logit_scale.exp().item())),plan_sha256=np.asarray(ph),teacher_sha256=np.asarray(signature['weights_sha256']))
    if candidates.exists() and a.teacher=='geoclip':
        with np.load(candidates) as z:
            if str(z['teacher_sha256'].item())!=signature['weights_sha256'] or str(z['plan_sha256'].item())!=ph: raise ValueError('Candidate cache provenance mismatch')
    if len(va):
        eval_names=[n for n in rows.filename if n in set(va.filename)]
        selected=va.set_index('filename').loc[eval_names]
        emb=read_embeddings(root,eval_names)
        if a.teacher=='geoclip':
            with np.load(candidates) as z: pred=z['latlng'][np.argmax(emb@z['embedding'].T,axis=1)]
        else:
            train_names=[n for n in tr.filename if cache_file(root,n).exists()]
            reference=read_embeddings(root,train_names); coords=tr.set_index('filename').loc[train_names,['lat','lng']].to_numpy()
            pred=np.concatenate([coords[np.argmax(emb[s:s+128]@reference.T,axis=1)] for s in range(0,len(emb),128)])
        result={**metrics(selected[['lat','lng']].to_numpy(),pred),'validation_rows':len(eval_names),'scope':'full fold' if a.all else 'preview subset only','plan_sha256':ph,'teacher':a.teacher,'candidate_type':'fine medoids' if a.teacher=='geoclip' else 'supplied train-image embeddings'}
        dest=Path(a.plan)/(a.teacher+('_full' if a.all else '_preview'))
        json_write(str(dest)+'_metrics.json',result)
        pd.DataFrame({'filename':eval_names,'pred_lat':pred[:,0],'pred_lng':pred[:,1]}).to_csv(str(dest)+'_predictions.csv',index=False)
        print(json.dumps(result,indent=2))

def targets(a):
    plan=plan_load(a.plan); ph=digest(Path(a.plan)/'plan.pt'); tr=pd.read_csv(Path(a.plan)/'train_rows.csv')
    arrays=dict(filename=tr.filename.to_numpy(dtype=str),plan_sha256=np.asarray(ph))
    provenance={}
    if a.geoclip:
        root=Path(a.geoclip); meta=json.loads((root/'teacher.json').read_text())
        image=read_embeddings(root,tr.filename)
        with np.load(Path(a.plan)/'geoclip_candidates.npz') as z:
            if str(z['plan_sha256'].item())!=ph or str(z['teacher_sha256'].item())!=meta['weights_sha256']: raise ValueError('KD candidate provenance mismatch')
            arrays['logits']=(image@z['embedding'].T*float(z['logit_scale'])).astype(np.float32)
        provenance['probability_teacher']=meta
    if a.features:
        raw=read_embeddings(a.features,tr.filename)
        dim=plan['model_config']['embedding_dim']; k=min(dim,len(raw)-1,raw.shape[1])
        if k<1: raise ValueError('Not enough train samples for PCA')
        pca=PCA(n_components=k,svd_solver='randomized',random_state=2026)
        projected=pca.fit_transform(raw)
        # PCA is fitted exclusively on this fold's TRAINING images.
        arrays['features']=norm(np.pad(projected,((0,0),(0,dim-k))))
        arrays['pca_mean']=pca.mean_; arrays['pca_components']=pca.components_
        provenance['feature_teacher']=json.loads((Path(a.features)/'teacher.json').read_text())
        provenance['pca_training_rows']=len(tr)
    if len(arrays)==2: raise ValueError('Supply --geoclip and/or --features')
    arrays['provenance']=np.asarray(json.dumps(provenance,sort_keys=True))
    npz_write(a.out,**arrays); print(f'Fold-specific targets saved: {a.out}')

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('cache'); q.add_argument('--teacher',choices=['geoclip','streetclip'],required=True); q.add_argument('--plan',required=True); q.add_argument('--data',required=True); q.add_argument('--cache',required=True); q.add_argument('--limit',type=int,default=256); q.add_argument('--all',action='store_true'); q.add_argument('--batch',type=int,default=4); q.add_argument('--device',default='cuda')
    q=sub.add_parser('targets'); q.add_argument('--plan',required=True); q.add_argument('--geoclip'); q.add_argument('--features'); q.add_argument('--out',required=True)
    a=p.parse_args(); globals()[a.command](a)

if __name__=='__main__': main()
