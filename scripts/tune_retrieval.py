#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geoguesser.checkpoint import load_checkpoint
from geoguesser.data import suggested_num_workers
from geoguesser.geo import metric_summary
from geoguesser.inference import extract_outputs
from geoguesser.model import decode_outputs
from geoguesser.retrieval import RetrievalConfig, hybrid_retrieval_predict
from geoguesser.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune train-only retrieval on a held-out fold")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-bank", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", choices=["none", "rot180", "rot4"], default="none")
    parser.add_argument("--acknowledge-reference-bank-rule", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_reference_bank_rule:
        raise SystemExit("Obtain instructor clarification before treating a reference bank as submission-safe.")
    device = choose_device(args.device)
    model, fine, _, bundle = load_checkpoint(args.checkpoint, device)
    frame = pd.read_csv(args.validation_csv)
    config = bundle["config"]
    filenames, output = extract_outputs(
        model,
        frame,
        args.image_dir,
        image_size=int(bundle["metadata"]["image_size"]),
        normalization=str(config["model"].get("normalization", "half")),
        batch_size=args.batch_size,
        num_workers=suggested_num_workers(args.num_workers),
        device=device,
        tta=args.tta,
    )
    if filenames.tolist() != frame["filename"].tolist():
        raise RuntimeError("Validation ordering mismatch")
    torch_output = {key: torch.from_numpy(value).to(device) for key, value in output.items()}
    base, _ = decode_outputs(
        torch_output,
        fine,
        top_cells=int(config["inference"]["top_cells"]),
        temperature=float(config["inference"]["temperature"]),
        country_gate=float(config["inference"]["country_gate"]),
        residual_scale=float(config["inference"]["residual_scale"]),
        global_blend=float(config["inference"]["global_blend"]),
    )
    truth = frame[["lat", "lng"]].to_numpy()
    rows = []
    with np.load(args.reference_bank, allow_pickle=False) as bank:
        for top_cells, neighbors, cell_weight, blend in itertools.product(
            (3, 5, 8), (1, 3, 5), (0.25, 0.5, 0.75), (0.5, 0.75, 1.0)
        ):
            retrieval_config = RetrievalConfig(
                top_cells=top_cells,
                neighbors=neighbors,
                embedding_temperature=0.08,
                cell_logprob_weight=cell_weight,
                retrieval_blend=blend,
            )
            prediction = hybrid_retrieval_predict(
                output["embedding"], output["fine_logits"], output["country_logits"],
                base.cpu().numpy(), bank["embedding"], bank["latlng"], bank["fine_cell"],
                fine.cell_country_ids_, float(config["inference"]["country_gate"]), retrieval_config,
            )
            rows.append({**retrieval_config.__dict__, **metric_summary(truth, prediction)})
    result = pd.DataFrame(rows).sort_values(["median_km", "mean_km"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
