"""Home-machine CPU work: oracle, frozen retrieval grids, exact CSV validation."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import aligned_labels, digest, json_write, ll, metrics, norm, validate_submission, xyz

def oracle(a):
    labels=aligned_labels(Path(a.data)/'train_labels.csv',a.folds)
    tr=labels[labels.fold!=a.fold]; va=labels[labels.fold==a.fold]
    bank=xyz(tr[['lat','lng']]); truth=va[['lat','lng']].to_numpy(); preds=[]
    for start in range(0,len(va),128): preds.append(tr[['lat','lng']].to_numpy()[np.argmax(xyz(truth[start:start+128])@bank.T,1)])
    result=metrics(truth,np.concatenate(preds)); result['note']='Oracle lower bound for selecting one training coordinate only; NOT model accuracy'
    json_write(a.out,result); print(json.dumps(result,indent=2))

def retrieval_predict(query,bank,config):
    coords=xyz(bank['truth']); emb=norm(bank['embedding']); q=norm(query['embedding']); predictions=[]
    for start in range(0,len(q),128):
        sim=q[start:start+128]@emb.T
        if config['country_gate']:
            allowed=query['country'][start:start+128,None]==bank['true_country'][None,:]
            # Fall back to unrestricted only if the predicted country has no bank member.
            allowed[~allowed.any(1)]=True
            sim=np.where(allowed,sim,-np.inf)
        k=min(config['k'],len(emb))
        ix=np.argsort(-sim,axis=1,kind='stable')[:,:k]
        score=np.take_along_axis(sim,ix,1)/config['temperature']; score-=score.max(1,keepdims=True)
        weights=np.exp(score); weights/=weights.sum(1,keepdims=True)
        pred=norm((coords[ix]*weights[...,None]).sum(1))
        alpha=config['blend']
        predictions.append(ll((1-alpha)*xyz(query['pred'][start:start+128])+alpha*pred))
    return np.concatenate(predictions)

def retrieval(a):
    with np.load(a.bank,allow_pickle=False) as z: bank={k:z[k] for k in z.files}
    with np.load(a.query,allow_pickle=False) as z: query={k:z[k] for k in z.files}
    tr=pd.read_csv(Path(a.plan)/'train_rows.csv'); va=pd.read_csv(Path(a.plan)/'validation_rows.csv')
    for cache,frame in ((bank,tr),(query,va)):
        names=cache['filename'].tolist()
        if len(set(names))!=len(names) or set(names)!=set(frame.filename): raise ValueError('Cache does not match complete plan rows')
        truth=frame.set_index('filename').loc[names,['lat','lng']].to_numpy()
        if not np.allclose(cache['truth'],truth,rtol=0,atol=1e-5): raise ValueError('Coordinates mismatch')
    if set(bank['filename'])&set(query['filename']): raise ValueError('Retrieval train/validation leakage')
    if str(bank['checkpoint_sha256'])!=str(query['checkpoint_sha256']): raise ValueError('Cache checkpoints differ')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if a.selected:
        settings=[json.loads(Path(a.selected).read_text())['config']]
    else:
        manifest=json.loads((Path(a.plan)/'manifest.json').read_text())
        if str(manifest['fold'])!='0': raise ValueError('Tune only fold 0; use --selected for confirmation')
        settings=[dict(k=k,temperature=t,country_gate=g,blend=b) for k,t,g,b in itertools.product((1,3,5,10),(.05,.1),(False,True),(0.,.25,.5,1.))]
    records=[]; best=None; best_pred=None
    for cfg in settings:
        prediction=retrieval_predict(query,bank,cfg)
        row={**cfg,**metrics(query['truth'],prediction)}; records.append(row)
        if best is None or (row['median_km'],row['mean_km'])<(best['median_km'],best['mean_km']): best=row; best_pred=prediction
    pd.DataFrame(records).sort_values(['median_km','mean_km']).to_csv(out/'retrieval_metrics.csv',index=False)
    cfg={k:best[k] for k in ('k','temperature','country_gate','blend')}
    if not a.selected: json_write(out/'selected.json',dict(config=cfg,source_fold=0,metrics=best))
    json_write(out/'comparison.json',dict(neural=metrics(query['truth'],query['pred']),retrieval=best))
    pd.DataFrame({'filename':query['filename'],'pred_lat':best_pred[:,0],'pred_lng':best_pred[:,1]}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps(best,indent=2))

def validate(a):
    validate_submission(pd.read_csv(a.csv),Path(a.data)/'holdout_public'); print('PASS: exact columns, finite/ranged coordinates, 2400 unique matching images')

def compare(a):
    labels=pd.read_csv(Path(a.plan)/'validation_rows.csv').set_index('filename')
    first=pd.read_csv(a.first).set_index('filename'); second=pd.read_csv(a.second).set_index('filename')
    if first.index.duplicated().any() or second.index.duplicated().any(): raise ValueError('Duplicate predictions')
    if not set(first.index)<=set(labels.index) or not set(second.index)<=set(labels.index): raise ValueError('Predictions are not validation rows')
    names=sorted(set(first.index)&set(second.index))
    if not names: raise ValueError('No common validation filenames')
    truth=labels.loc[names,['lat','lng']].to_numpy()
    print(json.dumps(dict(rows=len(names),first=metrics(truth,first.loc[names,['pred_lat','pred_lng']].to_numpy()),second=metrics(truth,second.loc[names,['pred_lat','pred_lng']].to_numpy())),indent=2))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('oracle'); q.add_argument('--data',required=True); q.add_argument('--folds',required=True); q.add_argument('--fold',type=int,default=0); q.add_argument('--out',required=True)
    q=sub.add_parser('retrieval'); q.add_argument('--plan',required=True); q.add_argument('--bank',required=True); q.add_argument('--query',required=True); q.add_argument('--selected'); q.add_argument('--out',required=True)
    q=sub.add_parser('validate'); q.add_argument('--csv',required=True); q.add_argument('--data',required=True)
    q=sub.add_parser('compare'); q.add_argument('--plan',required=True); q.add_argument('--first',required=True); q.add_argument('--second',required=True)
    a=p.parse_args(); globals()[a.command](a)

if __name__=='__main__': main()
