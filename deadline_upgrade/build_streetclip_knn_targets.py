import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_embeddings, digest, npz_write, metrics

p = argparse.ArgumentParser()
p.add_argument("--plan", required=True)
p.add_argument("--cache", required=True)
p.add_argument("--out", required=True)
p.add_argument("--k", type=int, default=8)
p.add_argument("--teacher-temperature", type=float, default=0.05)
p.add_argument("--block", type=int, default=512)
p.add_argument("--device", default="cuda")
a = p.parse_args()

plan = Path(a.plan)
tr = pd.read_csv(plan / "train_rows.csv")

print("Loading cached StreetCLIP embeddings...")
emb = read_embeddings(a.cache, tr.filename.tolist()).astype(np.float32)

device = torch.device(a.device)
E = torch.from_numpy(emb).to(device)
cells = torch.tensor(tr.fine_cell.to_numpy(), dtype=torch.long, device=device)

n = len(tr)
n_cells = int(tr.fine_cell.max()) + 1
k = min(a.k, n - 1)

teacher_logits = np.empty((n, n_cells), dtype=np.float32)
top1_indices = np.empty(n, dtype=np.int64)

print(f"Building teacher distributions: N={n}, cells={n_cells}, k={k}")

for start in range(0, n, a.block):
    end = min(start + a.block, n)

    sim = E[start:end] @ E.T

    local = torch.arange(end - start, device=device)
    global_idx = torch.arange(start, end, device=device)

    # Never let an image retrieve itself.
    sim[local, global_idx] = -1e9

    values, indices = torch.topk(sim, k=k, dim=1)

    top1_indices[start:end] = indices[:, 0].cpu().numpy()

    weights = torch.softmax(values / a.teacher_temperature, dim=1)

    probs = torch.zeros(
        (end - start, n_cells),
        dtype=torch.float32,
        device=device
    )

    neighbour_cells = cells[indices]
    probs.scatter_add_(1, neighbour_cells, weights)

    # student.py treats these as logits, then softmaxes them.
    teacher_logits[start:end] = torch.log(
        probs.clamp_min(1e-7)
    ).cpu().numpy()

    print(f"{end}/{n}", flush=True)

# Diagnostic only: self-excluded StreetCLIP nearest neighbour on TRAIN.
coords = tr[["lat", "lng"]].to_numpy()
top1_pred = coords[top1_indices]
diag = metrics(coords, top1_pred)

print("Self-excluded StreetCLIP train-neighbour diagnostic:")
print(json.dumps(diag, indent=2))

npz_write(
    a.out,
    filename=tr.filename.to_numpy(dtype=str),
    plan_sha256=np.asarray(digest(plan / "plan.pt")),
    logits=teacher_logits,
    provenance=np.asarray(json.dumps({
        "teacher": "StreetCLIP",
        "method": "top-k train-image neighbourhood -> fine-cell distribution",
        "k": k,
        "teacher_temperature": a.teacher_temperature,
        "self_excluded": True,
    }, sort_keys=True))
)

print("Saved:", a.out)
