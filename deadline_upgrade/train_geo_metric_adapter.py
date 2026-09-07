import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

R = 6371.0088


def normalize_np(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def haversine_np(a, b):
    lat1 = np.radians(a[:, 0])
    lon1 = np.radians(a[:, 1])
    lat2 = np.radians(b[:, 0])
    lon2 = np.radians(b[:, 1])

    h = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class ResidualMetric(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        # Starts exactly as the original embedding.
        return F.normalize(x + self.proj(x), dim=1)


@torch.inference_mode()
def evaluate(model, train_x, train_y, val_x, val_y, device):
    model.eval()

    bank = model(torch.from_numpy(train_x).to(device))
    query = model(torch.from_numpy(val_x).to(device))

    # Chunk queries to keep memory bounded.
    pred_parts = []

    for start in range(0, len(query), 512):
        q = query[start:start + 512]
        idx = (q @ bank.T).argmax(dim=1).cpu().numpy()
        pred_parts.append(train_y[idx])

    pred = np.concatenate(pred_parts)

    d = haversine_np(val_y, pred)

    return {
        "median_km": float(np.median(d)),
        "mean_km": float(np.mean(d)),
        "lt_200": float(np.mean(d < 200)),
        "lt_750": float(np.mean(d < 750)),
        "lt_40": float(np.mean(d < 40)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigmas", default="25,50,100,200")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pos-k", type=int, default=4)
    ap.add_argument("--neg-per-anchor", type=int, default=4)
    ap.add_argument("--neg-min-km", type=float, default=500.0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.train, allow_pickle=False) as z:
        train_x = normalize_np(z["embedding"].astype(np.float32))
        train_y = z["truth"].astype(np.float32)

    with np.load(args.val, allow_pickle=False) as z:
        val_x = normalize_np(z["embedding"].astype(np.float32))
        val_y = z["truth"].astype(np.float32)

    n, dim = train_x.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("train:", train_x.shape, "val:", val_x.shape)

    identity = ResidualMetric(dim).to(device)

    baseline = evaluate(
        identity, train_x, train_y, val_x, val_y, device
    )

    print("\nBASELINE")
    print(json.dumps(baseline, indent=2))

    # ---------------------------------------------------------
    # Positive pairs: geographic nearest neighbours.
    # ---------------------------------------------------------
    gps_rad = np.radians(train_y)

    nbrs = NearestNeighbors(
        n_neighbors=args.pos_k + 1,
        metric="haversine",
        algorithm="ball_tree",
    ).fit(gps_rad)

    pos_dist_rad, pos_idx = nbrs.kneighbors(gps_rad)

    pos_i = np.repeat(np.arange(n), args.pos_k)
    pos_j = pos_idx[:, 1:args.pos_k + 1].reshape(-1)
    pos_d = (
        pos_dist_rad[:, 1:args.pos_k + 1].reshape(-1) * R
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Negative pairs: random geographically distant images.
    # ---------------------------------------------------------
    neg_i = np.repeat(np.arange(n), args.neg_per_anchor)
    neg_j = np.random.randint(0, n, size=len(neg_i))

    for _ in range(20):
        d = haversine_np(train_y[neg_i], train_y[neg_j])
        bad = d < args.neg_min_km

        if not bad.any():
            break

        neg_j[bad] = np.random.randint(0, n, size=bad.sum())

    neg_d = haversine_np(
        train_y[neg_i], train_y[neg_j]
    ).astype(np.float32)

    pair_i = np.concatenate([pos_i, neg_i])
    pair_j = np.concatenate([pos_j, neg_j])
    pair_d = np.concatenate([pos_d, neg_d])

    print(
        f"pairs={len(pair_i)}, "
        f"positive={len(pos_i)}, negative={len(neg_i)}"
    )

    x_tensor = torch.from_numpy(train_x).to(device)

    results = []
    best_global = None

    for sigma in [float(x) for x in args.sigmas.split(",")]:
        print("\n====================================")
        print("SIGMA", sigma)
        print("====================================")

        model = ResidualMetric(dim).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )

        target = np.exp(-pair_d / sigma).astype(np.float32)

        best_sigma = None
        best_state = None

        for epoch in range(1, args.epochs + 1):
            model.train()

            order = np.random.permutation(len(pair_i))
            total = 0.0
            seen = 0

            for start in range(0, len(order), args.batch):
                ids = order[start:start + args.batch]

                ii = torch.from_numpy(pair_i[ids]).long().to(device)
                jj = torch.from_numpy(pair_j[ids]).long().to(device)

                y = torch.from_numpy(target[ids]).to(device)

                zi = model(x_tensor[ii])
                zj = model(x_tensor[jj])

                sim = (zi * zj).sum(dim=1)

                # Nearby examples matter more, but negatives still count.
                weight = 1.0 + 3.0 * y

                loss = (
                    weight * F.smooth_l1_loss(
                        sim, y, reduction="none"
                    )
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.item() * len(ids)
                seen += len(ids)

            metrics = evaluate(
                model, train_x, train_y, val_x, val_y, device
            )

            print(
                f"epoch={epoch:02d} "
                f"loss={total/seen:.5f} "
                f"median={metrics['median_km']:.3f} "
                f"mean={metrics['mean_km']:.3f}"
            )

            if (
                best_sigma is None
                or metrics["median_km"] < best_sigma["median_km"]
            ):
                best_sigma = dict(metrics)
                best_sigma["sigma"] = sigma
                best_sigma["epoch"] = epoch

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        results.append(best_sigma)

        if (
            best_global is None
            or best_sigma["median_km"] < best_global["median_km"]
        ):
            best_global = dict(best_sigma)
            torch.save(
                {
                    "state_dict": best_state,
                    "dim": dim,
                    "config": best_global,
                },
                out / "best_adapter.pt",
            )

    df = pd.DataFrame(results).sort_values("median_km")
    df.to_csv(out / "results.csv", index=False)

    with open(out / "best_config.json", "w") as f:
        json.dump(best_global, f, indent=2)

    print("\n====================================")
    print("FINAL RESULTS")
    print("====================================")
    print(df.to_string(index=False))

    print("\nBEST")
    print(json.dumps(best_global, indent=2))

    adapter_params = dim * dim

    print("\nadapter parameters:", adapter_params)
    print("final-model estimate:", 4_876_618 + adapter_params)


if __name__ == "__main__":
    main()
