import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


R = 6371.0088


def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def haversine(a, b):
    lat1 = np.radians(a[:, 0])
    lon1 = np.radians(a[:, 1])
    lat2 = np.radians(b[:, 0])
    lon2 = np.radians(b[:, 1])

    h = (
        np.sin((lat2-lat1)/2)**2
        + np.cos(lat1)*np.cos(lat2)
        * np.sin((lon2-lon1)/2)**2
    )

    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def xyz(latlon):
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])

    return np.stack([
        np.cos(lat)*np.cos(lon),
        np.cos(lat)*np.sin(lon),
        np.sin(lat),
    ], axis=1)


def xyz_to_latlon(v):
    v = normalize(v)

    lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))

    return np.stack([lat, lon], axis=1)


def spherical_weighted_mean(coords, weights):
    v = xyz(coords.reshape(-1, 2)).reshape(
        coords.shape[0], coords.shape[1], 3
    )

    out = (v * weights[..., None]).sum(axis=1)

    return xyz_to_latlon(out)


def metrics(truth, pred):
    d = haversine(truth, pred)

    return {
        "median_km": float(np.median(d)),
        "mean_km": float(np.mean(d)),
        "lt_200": float(np.mean(d < 200)),
        "lt_750": float(np.mean(d < 750)),
        "lt_40": float(np.mean(d < 40)),
    }


def topk_indices(scores, k):
    idx = np.argpartition(scores, -k, axis=1)[:, -k:]

    row = np.arange(len(scores))[:, None]
    vals = scores[row, idx]

    order = np.argsort(vals, axis=1)[:, ::-1]

    return idx[row, order], vals[row, order]


def evaluate_score_matrix(scores, bank_y, truth, prefix):
    rows = []

    # ---------------------------------------------------------
    # Pure top-1
    # ---------------------------------------------------------
    idx1 = scores.argmax(axis=1)
    pred1 = bank_y[idx1]

    m = metrics(truth, pred1)
    rows.append({
        "method": prefix + "_top1",
        **m
    })

    # ---------------------------------------------------------
    # Weighted spherical kNN
    # ---------------------------------------------------------
    for k in [3, 5, 10]:
        idx, vals = topk_indices(scores, k)
        coords = bank_y[idx]

        for temp in [0.02, 0.05, 0.10]:
            logits = (vals - vals[:, :1]) / temp
            w = np.exp(logits)
            w /= w.sum(axis=1, keepdims=True)

            pred = spherical_weighted_mean(coords, w)

            m = metrics(truth, pred)

            rows.append({
                "method": f"{prefix}_k{k}_t{temp}",
                **m
            })

    # ---------------------------------------------------------
    # Confidence-adaptive:
    # confident query -> top1
    # uncertain query -> weighted top3
    # ---------------------------------------------------------
    idx3, vals3 = topk_indices(scores, 3)

    logits = (vals3 - vals3[:, :1]) / 0.05
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)

    pred3 = spherical_weighted_mean(bank_y[idx3], w)

    margin = vals3[:, 0] - vals3[:, 1]

    for threshold in [0.0025, 0.005, 0.01, 0.02, 0.04]:
        pred = pred1.copy()

        uncertain = margin < threshold
        pred[uncertain] = pred3[uncertain]

        m = metrics(truth, pred)

        rows.append({
            "method": f"{prefix}_adaptive_{threshold}",
            **m
        })

    return rows


def similarity(q, b, chunk=512):
    parts = []

    for start in range(0, len(q), chunk):
        parts.append(q[start:start+chunk] @ b.T)

    return np.concatenate(parts, axis=0)


def csls_scores(q, b, k=10):
    sim = similarity(q, b)

    # Average similarity to local neighborhood.
    q_top = np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)

    # For each database point use query-set neighborhood.
    b_top = np.partition(sim, -k, axis=0)[-k:, :].mean(axis=0)

    return 2.0 * sim - q_top[:, None] - b_top[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.train, allow_pickle=False) as z:
        train_x0 = z["embedding"].astype(np.float32)
        train_y = z["truth"].astype(np.float32)

    with np.load(args.val, allow_pickle=False) as z:
        val_x0 = z["embedding"].astype(np.float32)
        truth = z["truth"].astype(np.float32)

    results = []

    # =========================================================
    # A. RAW
    # =========================================================
    train_x = normalize(train_x0)
    val_x = normalize(val_x0)

    print("RAW COSINE")
    raw = similarity(val_x, train_x)

    results += evaluate_score_matrix(
        raw, train_y, truth, "raw"
    )

    # =========================================================
    # B. MEAN CENTERING
    # Fit using TRAIN fold only.
    # =========================================================
    mean = train_x0.mean(axis=0, keepdims=True)

    train_c = normalize(train_x0 - mean)
    val_c = normalize(val_x0 - mean)

    print("CENTERED")
    centered = similarity(val_c, train_c)

    results += evaluate_score_matrix(
        centered, train_y, truth, "center"
    )

    # =========================================================
    # C. PCA / WHITENING
    # Fit using TRAIN fold only.
    # =========================================================
    for dim in [64, 128, 192]:
        print("PCA WHITEN", dim)

        pca = PCA(
            n_components=dim,
            whiten=True,
            random_state=42
        )

        train_p = pca.fit_transform(train_x0)
        val_p = pca.transform(val_x0)

        train_p = normalize(train_p.astype(np.float32))
        val_p = normalize(val_p.astype(np.float32))

        sim = similarity(val_p, train_p)

        results += evaluate_score_matrix(
            sim, train_y, truth,
            f"pca{dim}"
        )

    # =========================================================
    # D. CSLS
    # Hubness-corrected retrieval.
    # =========================================================
    print("CSLS RAW")
    csls = csls_scores(val_x, train_x, k=10)

    results += evaluate_score_matrix(
        csls, train_y, truth,
        "csls10"
    )

    print("CSLS CENTERED")
    csls_c = csls_scores(val_c, train_c, k=10)

    results += evaluate_score_matrix(
        csls_c, train_y, truth,
        "center_csls10"
    )

    # =========================================================
    # E. Average Query Expansion
    # =========================================================
    for qe_k in [2, 3, 5]:
        print("QUERY EXPANSION", qe_k)

        idx, _ = topk_indices(raw, qe_k)

        expanded = (
            val_x
            + train_x[idx].mean(axis=1)
        )

        expanded = normalize(expanded)

        sim = similarity(expanded, train_x)

        results += evaluate_score_matrix(
            sim, train_y, truth,
            f"qe{qe_k}"
        )

    df = pd.DataFrame(results)
    df = df.sort_values(
        ["median_km", "mean_km"],
        ascending=[True, True]
    )

    df.to_csv(out / "results.csv", index=False)

    best = df.iloc[0].to_dict()

    with open(out / "best.json", "w") as f:
        json.dump(best, f, indent=2)

    print()
    print("==========================================")
    print("TOP 20")
    print("==========================================")
    print(df.head(20).to_string(index=False))

    print()
    print("==========================================")
    print("BEST")
    print("==========================================")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
