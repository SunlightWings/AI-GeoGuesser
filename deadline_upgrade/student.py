"""Additive training/prepare/predict commands; existing core files remain unchanged."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from common import VERSION, aligned_labels, digest, json_write, metrics, npz_write, validate_submission
from geoguesser.cells import CountryKMeansPartition
from geoguesser.data import GeoDataset, add_country_ids
from geoguesser.model import build_model, model_config_from_runtime, model_parameter_count, decode_outputs
from geoguesser.checkpoint import make_checkpoint_bundle, load_checkpoint
from geoguesser.losses import GeoCriterion
from geoguesser.utils import atomic_torch_save, seed_everything

def load(path): return torch.load(path,map_location='cpu',weights_only=False)

def count_check(model):
    n=sum(p.numel() for p in model.parameters()) # includes requires_grad=False
    if n>5000000: raise ValueError(f'Inference model over cap: {n:,}')
    return n

def prepare(a):
    cfg=json.loads(Path(a.config).read_text())
    labels,mapping=add_country_ids(aligned_labels(Path(a.data)/'train_labels.csv',a.folds))
    train=labels if a.fold=='full' else labels[labels.fold!=int(a.fold)]
    val=labels.iloc[:0] if a.fold=='full' else labels[labels.fold==int(a.fold)]
    train=train.copy().reset_index(drop=True); val=val.copy().reset_index(drop=True)
    fine=CountryKMeansPartition(cfg['geocells']['fine_target_size'],cfg['seed'])
    coarse=CountryKMeansPartition(cfg['geocells']['coarse_target_size'],cfg['seed'])
    train['fine_cell']=fine.fit_transform(train); train['coarse_cell']=coarse.fit_transform(train)
    if len(val):
        val['fine_cell']=fine.transform(val); val['coarse_cell']=coarse.transform(val)
    mc=model_config_from_runtime(cfg,len(mapping),fine.n_cells_,coarse.n_cells_)
    n=count_check(build_model(mc,load_external_weights=False))
    out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Plan output already exists: reuse it, do not refit')
    out.mkdir(parents=True,exist_ok=True)
    train.to_csv(out/'train_rows.csv',index=False); val.to_csv(out/'validation_rows.csv',index=False)
    meta=dict(version=VERSION,fold=a.fold,folds_sha256=digest(a.folds),labels_sha256=digest(Path(a.data)/'train_labels.csv'),parameter_count=n)
    atomic_torch_save(dict(config=cfg,model_config=mc,fine=fine.to_state(),coarse=coarse.to_state(),country_to_id=mapping,metadata=meta),out/'plan.pt')
    json_write(out/'manifest.json',{**meta,'plan_sha256':digest(out/'plan.pt')})
    print(json.dumps(meta,indent=2))

def transform(size,train):
    ops=[T.Resize((size,size),interpolation=InterpolationMode.BICUBIC)]
    if train: ops += [T.ColorJitter(.15,.15,.10,.02)]
    return T.Compose(ops+[T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])

def loader(frame,data,size,batch,workers,train,epoch=0,seed=2026):
    # Full-image view shared with cached targets; no rotations, crops or mirroring.
    ds=GeoDataset(frame,Path(data)/'train',transform(size,train),with_targets=True)
    gen=torch.Generator().manual_seed(seed+epoch)
    return DataLoader(ds,batch_size=batch,shuffle=train,num_workers=workers,pin_memory=True,generator=gen)

@torch.inference_mode()
def evaluate(model,dl,fine,cfg,device):
    model.eval(); names=[]; pred=[]; truth=[]; embeddings=[]; countries=[]; logits=[]
    for batch in dl:
        out=model(batch['image'].to(device))
        coord,_=decode_outputs(out,fine,**{k:cfg['inference'][k] for k in ('top_cells','temperature','country_gate','residual_scale','global_blend')})
        names.extend(batch['filename']); pred.append(coord.cpu().numpy()); truth.append(batch['latlng'].numpy())
        embeddings.append(out['embedding'].cpu().numpy()); countries.append(out['country_logits'].argmax(1).cpu().numpy())
        logits.append(out['fine_logits'].cpu().numpy())
    arrays=dict(filename=np.asarray(names),pred=np.concatenate(pred),truth=np.concatenate(truth),embedding=np.concatenate(embeddings),country=np.concatenate(countries),fine_logits=np.concatenate(logits))
    return metrics(arrays['truth'],arrays['pred']),arrays

def train(a):
    plan=load(Path(a.plan)/'plan.pt'); cfg=copy.deepcopy(plan['config']); ph=digest(Path(a.plan)/'plan.pt')
    if digest(Path(a.data)/'train_labels.csv')!=plan['metadata']['labels_sha256']: raise ValueError('Dataset labels differ from the shared plan; copy the same labels CSV')
    cfg['train']['stages'][0].update(epochs=a.epochs,image_size=a.size,batch_size=a.batch)
    cfg['kd'].update(probability_weight=a.kd_weight,feature_weight=a.feature_weight,temperature=a.temperature)
    seed=cfg['seed']; seed_everything(seed,True)
    fine=CountryKMeansPartition.from_state(plan['fine']); coarse=CountryKMeansPartition.from_state(plan['coarse'])
    tr=pd.read_csv(Path(a.plan)/'train_rows.csv'); va=pd.read_csv(Path(a.plan)/'validation_rows.csv')
    outdir=Path(a.out)
    if outdir.exists() and any(outdir.iterdir()) and not a.resume: raise ValueError('Run exists; use --resume or a new output directory')
    outdir.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device)
    model=build_model(plan['model_config'],load_external_weights=not a.resume).to(device)
    n=count_check(model)
    target=None; target_hash=None
    if a.variant!='A':
        if not a.targets: raise ValueError('KD requires --targets')
        target_hash=digest(a.targets)
        with np.load(a.targets,allow_pickle=False) as z:
            if str(z['plan_sha256'].item())!=ph: raise ValueError('KD targets have different fold/cells')
            if z['filename'].tolist()!=tr.filename.tolist(): raise ValueError('KD rows mismatch')
            target={k:torch.from_numpy(z[k].astype(np.float32)).to(device) for k in ('logits','features') if k in z}
        if a.variant in ('B','D') and 'logits' not in target: raise ValueError('GeoCLIP probability targets missing')
        if a.variant in ('C','D') and 'features' not in target: raise ValueError('PCA feature targets missing')
    rowmap={name:i for i,name in enumerate(tr.filename)}
    backbone=list(model.backbone.parameters()); ids={id(p) for p in backbone}
    optim=torch.optim.AdamW([{'params':backbone,'lr':cfg['train']['backbone_lr']},{'params':[p for p in model.parameters() if id(p) not in ids],'lr':cfg['train']['head_lr']}],weight_decay=cfg['train']['weight_decay'])
    # Fixed 24-epoch schedule permits fair 8-epoch screens followed by exact continuation.
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda e:min(1.,(e+1)/2)*(.1+.9*(1+np.cos(np.pi*min(e,24)/24))/2))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    criterion=GeoCriterion(fine,coarse,cfg).to(device)
    start=0; best=(float('inf'),float('inf')); history=[]
    if a.confirm and str(plan['metadata']['fold'])!='1': raise ValueError('--confirm is for untouched fold 1')
    runmeta=dict(version=VERSION,plan_sha256=ph,variant=a.variant,targets_sha256=target_hash,size=a.size,batch=a.batch,kd_weight=a.kd_weight,feature_weight=a.feature_weight,temperature=a.temperature,confirm=a.confirm,confirmation_epochs=a.epochs if a.confirm else None)
    if a.resume:
        r=load(outdir/'last_training.pt')
        if r['runmeta']!=runmeta: raise ValueError('Resume config/targets changed; create a separate run')
        model.load_state_dict(r['model']); optim.load_state_dict(r['optimizer']); scheduler.load_state_dict(r['scheduler']); scaler.load_state_dict(r['scaler'])
        start=r['epoch']+1; best=tuple(r['best']); history=r['history']
        random.setstate(r['random']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(r['cuda_rng'])
    json_write(outdir/'config.json',cfg); json_write(outdir/'run.json',runmeta)
    tr.to_csv(outdir/'train_rows.csv',index=False); va.to_csv(outdir/'validation_rows.csv',index=False)
    for epoch in range(start,a.epochs):
        tick=time.time(); model.train(); total=0.; batches=0
        dl=loader(tr,a.data,a.size,a.batch,a.workers,True,epoch,seed)
        for batch in dl:
            x={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                output=model(x['image'])
            # Geographic trigonometry and KL in float32 even with mixed-precision backbone.
            output={k:v.float() for k,v in output.items()}
            loss,_=criterion(output,x)
            if target is not None:
                ix=torch.tensor([rowmap[s] for s in x['filename']],device=device)
                if a.variant in ('B','D'):
                    temp=a.temperature
                    kd=F.kl_div(F.log_softmax(output['fine_logits']/temp,1),F.softmax(target['logits'][ix]/temp,1),reduction='batchmean')*temp**2
                    loss=loss+a.kd_weight*kd
                if a.variant in ('C','D'):
                    loss=loss+a.feature_weight*(1-F.cosine_similarity(output['embedding'],target['features'][ix],dim=1)).mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; stopping without overwriting best checkpoint')
            scaler.scale(loss).backward(); scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optim); scaler.update(); total+=loss.item(); batches+=1
        scheduler.step()
        record=dict(epoch=epoch+1,train_loss=total/max(batches,1))
        arrays=None
        if len(va):
            met,arrays=evaluate(model,loader(va,a.data,a.size,a.batch,a.workers,False),fine,cfg,device)
            record.update(met)
        record['seconds']=time.time()-tick; history.append(record)
        metadata={**plan['metadata'],**runmeta,'image_size':a.size,'epoch':epoch+1,'normalization':'imagenet'}
        bundle=make_checkpoint_bundle(model,plan['model_config'],fine,coarse,plan['country_to_id'],cfg,metadata)
        bundle['rule_compliant']=True; bundle['rule_basis']='TA clarification: pretrained student permitted; teacher training-only'
        if not len(va) or a.confirm or (record['median_km'],record['mean_km'])<best:
            if len(va): best=(record['median_km'],record['mean_km'])
            atomic_torch_save(bundle,outdir/('best_model.pt' if len(va) else 'final_model.pt'))
            if arrays is not None:
                pd.DataFrame({'filename':arrays['filename'],'pred_lat':arrays['pred'][:,0],'pred_lng':arrays['pred'][:,1]}).to_csv(outdir/'oof_predictions.csv',index=False)
                npz_write(outdir/'validation.npz',**arrays,checkpoint_sha256=np.asarray(digest(outdir/'best_model.pt')))
                json_write(outdir/'metrics.json',record)
        pd.DataFrame(history).to_csv(outdir/'history.csv',index=False)
        atomic_torch_save(dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,history=history,runmeta=runmeta,random=random.getstate(),numpy=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),outdir/'last_training.pt')
        print(json.dumps(record),flush=True)
    print(f'Inference parameters: {n:,}; outputs: {outdir}')

def predict(a):
    device=torch.device(a.device); model,fine,_,bundle=load_checkpoint(a.checkpoint,device)
    count_check(model)
    if bundle['metadata'].get('version')!=VERSION: raise ValueError('Use the existing inference pipeline for older checkpoints; this command expects a deadline-upgrade model')
    if bundle['metadata'].get('fold')!='full': raise ValueError('Final submission requires the full-data checkpoint')
    from geoguesser.inference import predict_frame
    root=Path(a.data)/'holdout_public'
    names=sorted(p.name for p in root.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    cfg=bundle['config']
    names,pred=predict_frame(model,fine,pd.DataFrame({'filename':names}),root,cfg['inference'],image_size=bundle['metadata']['image_size'],normalization='imagenet',batch_size=a.batch,num_workers=a.workers,device=device,tta='none')
    frame=pd.DataFrame({'filename':names,'pred_lat':pred[:,0],'pred_lng':pred[:,1]})
    validate_submission(frame,root)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); frame.to_csv(a.out,index=False)
    print('Validated: 2400 unique holdout predictions; no teacher loaded.')

def export(a):
    device=torch.device(a.device); model,fine,_,bundle=load_checkpoint(a.checkpoint,device); count_check(model)
    if bundle['metadata'].get('version')!=VERSION: raise ValueError('Export expects a new ImageNet-normalized deadline-upgrade model, not a scratch baseline')
    frame=pd.read_csv(a.rows)
    _,arrays=evaluate(model,loader(frame,a.data,bundle['metadata']['image_size'],a.batch,a.workers,False),fine,bundle['config'],device)
    npz_write(a.out,**arrays,true_country=frame.country_id.to_numpy(),checkpoint_sha256=np.asarray(digest(a.checkpoint)))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--data',required=True); q.add_argument('--folds',required=True); q.add_argument('--fold',choices=['0','1','2','3','4','full'],required=True); q.add_argument('--out',required=True); q.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    q=sub.add_parser('train'); q.add_argument('--plan',required=True); q.add_argument('--data',required=True); q.add_argument('--out',required=True); q.add_argument('--variant',choices=list('ABCD'),default='A'); q.add_argument('--targets'); q.add_argument('--epochs',type=int,default=24); q.add_argument('--size',type=int,default=320); q.add_argument('--batch',type=int,default=16); q.add_argument('--workers',type=int,default=2); q.add_argument('--device',default='cuda'); q.add_argument('--resume',action='store_true'); q.add_argument('--confirm',action='store_true',help='Fold 1: use fixed last epoch, not validation-selected best'); q.add_argument('--kd-weight',type=float,default=.3); q.add_argument('--feature-weight',type=float,default=.3); q.add_argument('--temperature',type=float,default=2.)
    for name in ('predict','export'):
        q=sub.add_parser(name); q.add_argument('--checkpoint',required=True); q.add_argument('--data',required=True); q.add_argument('--out',required=True); q.add_argument('--device',default='cuda'); q.add_argument('--batch',type=int,default=16); q.add_argument('--workers',type=int,default=2)
        if name=='export': q.add_argument('--rows',required=True)
    a=p.parse_args()
    if hasattr(a,'temperature') and a.temperature<=0: p.error('temperature must be positive')
    globals()[a.command](a)

if __name__=='__main__': main()
