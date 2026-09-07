import numpy as np
from pathlib import Path

R = 6371.0088

def norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

def hav(a, b):
    la1, lo1 = np.radians(a[:,0]), np.radians(a[:,1])
    la2, lo2 = np.radians(b[:,0]), np.radians(b[:,1])
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))

def metric(y, p):
    d = hav(y,p)
    return (
        np.median(d), np.mean(d),
        100*np.mean(d<200),
        100*np.mean(d<750),
        100*np.mean(d<40)
    )

def bank_penalty(b, k=10, chunk=512):
    n = len(b)
    out = np.empty(n, np.float32)

    for s in range(0,n,chunk):
        e = min(s+chunk,n)
        sim = b[s:e] @ b.T

        # Remove self similarity.
        rows = np.arange(e-s)
        cols = np.arange(s,e)
        sim[rows,cols] = -np.inf

        top = np.partition(sim,-k,axis=1)[:,-k:]
        out[s:e] = top.mean(axis=1)

    return out

truth_all, raw_all, corrected_all = [], [], []

for f in range(5):
    run = Path(f"deadline/runs/CV_fold{f}")

    with np.load(run/"train_bank.npz", allow_pickle=False) as z:
        b = norm(z["embedding"].astype(np.float32))
        by = z["truth"].astype(np.float32)

    with np.load(run/"validation.npz", allow_pickle=False) as z:
        q = norm(z["embedding"].astype(np.float32))
        y = z["truth"].astype(np.float32)

    penalty = bank_penalty(b, k=10)

    raw_pred = []
    cor_pred = []

    for s in range(0,len(q),512):
        sim = q[s:s+512] @ b.T

        raw_idx = sim.argmax(axis=1)

        # Inductive CSLS-equivalent ranking:
        # r_query is constant across candidates, so omit it.
        corrected = 2*sim - penalty[None,:]
        cor_idx = corrected.argmax(axis=1)

        raw_pred.append(by[raw_idx])
        cor_pred.append(by[cor_idx])

    raw_pred = np.concatenate(raw_pred)
    cor_pred = np.concatenate(cor_pred)

    print("fold",f)
    print(" raw :", metric(y,raw_pred))
    print(" bank:", metric(y,cor_pred))

    truth_all.append(y)
    raw_all.append(raw_pred)
    corrected_all.append(cor_pred)

truth = np.concatenate(truth_all)
raw = np.concatenate(raw_all)
corrected = np.concatenate(corrected_all)

print("\n===== POOLED =====")
print("RAW :", metric(truth,raw))
print("BANK:", metric(truth,corrected))
