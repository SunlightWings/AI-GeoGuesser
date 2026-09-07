import numpy as np
from pathlib import Path

R = 6371.0088

def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

def haversine(a, b):
    lat1 = np.radians(a[:,0]); lon1 = np.radians(a[:,1])
    lat2 = np.radians(b[:,0]); lon2 = np.radians(b[:,1])
    h = (
        np.sin((lat2-lat1)/2)**2
        + np.cos(lat1)*np.cos(lat2)
        * np.sin((lon2-lon1)/2)**2
    )
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))

def metrics(truth, pred):
    d = haversine(truth, pred)
    return (
        float(np.median(d)),
        float(np.mean(d)),
        100*float(np.mean(d < 200)),
        100*float(np.mean(d < 750)),
        100*float(np.mean(d < 40)),
    )

truth_all = []
raw_all = []
csls_all = []

for f in range(5):
    run = Path(f"deadline/runs/CV_fold{f}")

    with np.load(run/"train_bank.npz", allow_pickle=False) as z:
        b = normalize(z["embedding"].astype(np.float32))
        by = z["truth"].astype(np.float32)

    with np.load(run/"validation.npz", allow_pickle=False) as z:
        q = normalize(z["embedding"].astype(np.float32))
        truth = z["truth"].astype(np.float32)

    sim = q @ b.T

    raw_idx = sim.argmax(axis=1)
    raw_pred = by[raw_idx]

    k = 10
    q_local = np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)
    b_local = np.partition(sim, -k, axis=0)[-k:, :].mean(axis=0)
    csls = 2*sim - q_local[:,None] - b_local[None,:]

    csls_idx = csls.argmax(axis=1)
    csls_pred = by[csls_idx]

    print("fold", f)
    print(" raw :", metrics(truth, raw_pred))
    print(" csls:", metrics(truth, csls_pred))

    truth_all.append(truth)
    raw_all.append(raw_pred)
    csls_all.append(csls_pred)

truth = np.concatenate(truth_all)
raw = np.concatenate(raw_all)
csls = np.concatenate(csls_all)

print("\n===== POOLED =====")
print("RAW  median mean <200 <750 <40")
print(metrics(truth, raw))
print("CSLS median mean <200 <750 <40")
print(metrics(truth, csls))
