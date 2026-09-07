import numpy as np
from pathlib import Path


def normalize(x):
    return x / np.clip(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12,
        None
    )


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    d = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2.0) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


truth_all = []
pred_all = []

for fold in range(5):
    run = Path(f"deadline/runs/CV_fold{fold}")

    with np.load(run / "train_bank.npz", allow_pickle=False) as z:
        bank_emb = normalize(z["embedding"].astype(np.float32))
        bank_truth = z["truth"].astype(np.float32)

    with np.load(run / "validation.npz", allow_pickle=False) as z:
        query_emb = normalize(z["embedding"].astype(np.float32))
        truth = z["truth"].astype(np.float32)

    # Frozen inference configuration:
    # k=1, cosine similarity, no country gate, blend=1.
    similarity = query_emb @ bank_emb.T
    nearest = similarity.argmax(axis=1)
    pred = bank_truth[nearest]

    print(
        f"fold {fold}: val={len(truth)}"
    )

    truth_all.append(truth)
    pred_all.append(pred)


truth = np.concatenate(truth_all)
pred = np.concatenate(pred_all)

assert len(truth) == 11758, f"Expected 11758 OOF rows, got {len(truth)}"

d = haversine(
    truth[:, 0],
    truth[:, 1],
    pred[:, 0],
    pred[:, 1],
)

print()
print("===== POOLED 5-FOLD OOF RESULTS =====")
print(f"rows:       {len(d)}")
print(f"median_km:  {np.median(d):.6f}")
print(f"mean_km:    {np.mean(d):.6f}")
print(f"lt_200_pct: {100*np.mean(d < 200):.4f}")
print(f"lt_750_pct: {100*np.mean(d < 750):.4f}")
print(f"lt_40_pct:  {100*np.mean(d < 40):.4f}")
