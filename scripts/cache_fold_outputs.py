#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geoguesser.checkpoint import load_checkpoint
from geoguesser.data import suggested_num_workers
from geoguesser.inference import extract_outputs
from geoguesser.utils import choose_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache fold outputs once for decoder and frozen-embedding ablations"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--tta", nargs="+", choices=["none", "rot180", "rot4"], default=["none", "rot180"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    frame = pd.read_csv(args.validation_csv)
    required = {"filename", "lat", "lng"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Validation CSV is missing columns: {sorted(missing)}")
    device = choose_device(args.device)
    model, fine, coarse, bundle = load_checkpoint(args.checkpoint, device)
    config = bundle["config"]
    image_size = int(
        bundle["metadata"].get("image_size", config["train"]["stages"][-1]["image_size"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(args.checkpoint)

    for tta in args.tta:
        filenames, output = extract_outputs(
            model,
            frame,
            args.image_dir,
            image_size=image_size,
            normalization=str(config["model"].get("normalization", "half")),
            batch_size=args.batch_size,
            num_workers=suggested_num_workers(args.num_workers),
            device=device,
            tta=tta,
        )
        if filenames.tolist() != frame["filename"].tolist():
            raise RuntimeError("Cached output ordering differs from validation CSV")
        metadata = {
            "format_version": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "validation_csv": str(args.validation_csv.resolve()),
            "rows": len(frame),
            "tta": tta,
            "image_size": image_size,
        }
        destination = args.output_dir / f"validation_{tta}.npz"
        np.savez_compressed(
            destination,
            filename=filenames,
            truth_latlng=frame[["lat", "lng"]].to_numpy(dtype=np.float32),
            fine_centers_xyz=fine.output_centers_xyz_.astype(np.float32),
            fine_cell_country_ids=fine.cell_country_ids_.astype(np.int16),
            coarse_cell_country_ids=coarse.cell_country_ids_.astype(np.int16),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **{key: value.astype(np.float32) for key, value in output.items()},
        )
        print(json.dumps({"output": str(destination.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
