#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from geoguesser.checkpoint import load_checkpoint
from geoguesser.data import GeoDataset, build_transforms, suggested_num_workers
from geoguesser.geo import metric_summary
from geoguesser.inference import forward_tta
from geoguesser.model import decode_outputs
from geoguesser.utils import choose_device


@torch.inference_mode()
def collect_outputs(model, loader, device, tta):
    collected = None
    filenames = []
    for batch in tqdm(loader, desc=f"extract {tta}"):
        output = forward_tta(model, batch["image"].to(device), tta)
        if collected is None:
            collected = {key: [] for key in output}
        for key, value in output.items():
            collected[key].append(value.cpu())
        filenames.extend(batch["filename"])
    assert collected is not None
    return filenames, {key: torch.cat(value).to(device) for key, value in collected.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search decoder settings on one validation fold")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", default="inference_tuning.csv", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", nargs="+", choices=["none", "rot180", "rot4"], default=["none", "rot180"])
    args = parser.parse_args()
    device = choose_device(args.device)
    model, fine, _, bundle = load_checkpoint(args.checkpoint, device)
    config = bundle["config"]
    frame = pd.read_csv(args.validation_csv)
    dataset = GeoDataset(
        frame,
        args.image_dir,
        build_transforms(
            int(bundle["metadata"]["image_size"]),
            train=False,
            normalization=str(config["model"].get("normalization", "half")),
        ),
        with_targets=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=suggested_num_workers(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    truth = frame[["lat", "lng"]].to_numpy()
    rows = []
    for tta in args.tta:
        filenames, output = collect_outputs(model, loader, device, tta)
        if filenames != frame["filename"].tolist():
            raise RuntimeError("Validation ordering mismatch")
        for top_cells, temperature, gate, residual_scale, blend in itertools.product(
            (1, 3, 5, 8),
            (0.65, 0.85, 1.0, 1.2),
            (0.5, 1.0, 2.0, 4.0),
            (0.0, 0.01, 0.025, 0.05),
            (0.0, 0.05, 0.10),
        ):
            prediction, _ = decode_outputs(
                output,
                fine,
                top_cells=top_cells,
                temperature=temperature,
                country_gate=gate,
                residual_scale=residual_scale,
                global_blend=blend,
            )
            rows.append(
                {
                    "tta": tta,
                    "top_cells": top_cells,
                    "temperature": temperature,
                    "country_gate": gate,
                    "residual_scale": residual_scale,
                    "global_blend": blend,
                    **metric_summary(truth, prediction.cpu().numpy()),
                }
            )
    result = pd.DataFrame(rows).sort_values(["median_km", "mean_km"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(15).to_string(index=False))
    print("Best settings:", json.dumps(result.iloc[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
