#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from geoguesser.checkpoint import load_checkpoint
from geoguesser.data import add_country_ids, read_labels, suggested_num_workers
from geoguesser.inference import extract_outputs
from geoguesser.utils import choose_device


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only embedding reference bank")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="Use train_rows.csv for a fold or full train_labels.csv for final")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--acknowledge-reference-bank-rule", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_reference_bank_rule:
        raise SystemExit(
            "A reference bank is non-parametric and train-only, but the course may count it as part "
            "of the submitted system. Obtain clarification, then pass --acknowledge-reference-bank-rule."
        )
    device = choose_device(args.device)
    model, fine, _, bundle = load_checkpoint(args.checkpoint, device)
    frame, _ = add_country_ids(read_labels(args.labels), bundle["country_to_id"])
    frame["fine_cell"] = fine.transform(frame)
    missing = [name for name in frame["filename"] if not (args.image_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference images: {missing[:5]}")
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
    )
    metadata = {
        "format_version": 1,
        "source": "supplied labelled training images only",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "rows": len(frame),
        "embedding_dim": output["embedding"].shape[1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        filename=filenames,
        embedding=output["embedding"].astype(np.float16),
        latlng=frame[["lat", "lng"]].to_numpy(dtype=np.float32),
        fine_cell=frame["fine_cell"].to_numpy(dtype=np.int16),
        country_id=frame["country_id"].to_numpy(dtype=np.int8),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps({"output": str(args.output.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
