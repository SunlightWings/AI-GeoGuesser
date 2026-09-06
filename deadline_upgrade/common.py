"""Portable, dependency-light artifact contracts. No teacher imports here."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'deadline-v1'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)

def npz_write(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)

def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def xyz(ll):
    a,b=np.radians(np.asarray(ll,dtype=np.float64)).T
    return np.stack([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)],-1)

def ll(x):
    x=norm(x)
    return np.stack([np.degrees(np.arcsin(np.clip(x[...,2],-1,1))),np.degrees(np.arctan2(x[...,1],x[...,0]))],-1)

def metrics(truth,pred):
    truth=np.asarray(truth,dtype=float); pred=np.asarray(pred,dtype=float)
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1]!=2 or not len(truth):
        raise ValueError('Expected matching nonempty N x 2 coordinates')
    if not np.isfinite(truth).all() or not np.isfinite(pred).all(): raise ValueError('Nonfinite coordinates')
    a,b=np.radians(truth).T; c,d=np.radians(pred).T
    h=np.sin((a-c)/2)**2+np.cos(a)*np.cos(c)*np.sin((b-d)/2)**2
    e=12742.0176*np.arcsin(np.sqrt(np.clip(h,0,1)))
    return dict(median_km=float(np.median(e)),mean_km=float(e.mean()),lt_200_km=float((e<200).mean()),lt_750_km=float((e<750).mean()),lt_40_km=float((e<40).mean()))

def aligned_labels(labels_path, folds_path):
    labels=pd.read_csv(labels_path); folds=pd.read_csv(folds_path)
    for df in (labels,folds):
        if df.filename.duplicated().any() or df.filename.isna().any(): raise ValueError('Duplicate/missing filenames')
        if any(Path(n).name != n or '\\' in n for n in df.filename): raise ValueError('Unsafe filename')
    if set(labels.filename)!=set(folds.filename): raise ValueError('Folds must cover exactly the supplied labels')
    folds=folds.set_index('filename').loc[labels.filename]
    if not set(folds['fold'])==set(range(5)): raise ValueError('Expected canonical folds 0..4')
    for col in ('lat','lng'):
        if col in folds and not np.allclose(labels[col],folds[col],atol=1e-6,rtol=0): raise ValueError('Fold coordinates disagree')
    labels['fold']=folds['fold'].to_numpy()
    if not np.isfinite(labels[['lat','lng']]).all().all(): raise ValueError('Invalid labels')
    return labels.sort_values('filename').reset_index(drop=True)

def validate_submission(frame, image_dir, count=2400):
    if list(frame.columns)!=['filename','pred_lat','pred_lng']: raise ValueError('Exact header required: filename,pred_lat,pred_lng')
    names={p.name for p in Path(image_dir).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
    if len(names)!=count or len(frame)!=count or frame.filename.duplicated().any() or set(frame.filename)!=names:
        raise ValueError('Holdout must contain exactly 2400 unique matching filenames')
    if not np.isfinite(frame[['pred_lat','pred_lng']]).all().all(): raise ValueError('Nonfinite predictions')
    if not frame.pred_lat.between(-90,90).all() or not frame.pred_lng.between(-180,180).all(): raise ValueError('Coordinates out of range')
    return True

def cache_file(root,name):
    return Path(root)/'images'/(hashlib.sha256(name.encode()).hexdigest()+'.npz')

def read_embeddings(root,names):
    result=[]
    for name in names:
        with np.load(cache_file(root,name),allow_pickle=False) as z:
            if str(z['filename'].item())!=name: raise ValueError('Cache key mismatch')
            result.append(z['embedding'])
    return norm(np.stack(result))
