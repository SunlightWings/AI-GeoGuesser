#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoguesser.checkpoint import load_checkpoint
from geoguesser.data import read_labels, suggested_num_workers
from geoguesser.geo import metric_summary
from geoguesser.inference import predict_frame
from geoguesser.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on labelled images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", choices=["none", "rot180", "rot4"], default="none")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, fine, _, bundle = load_checkpoint(args.checkpoint, device)
    config = bundle["config"]
    frame = read_labels(args.labels)
    filenames, prediction = predict_frame(
        model,
        fine,
        frame,
        Path(args.image_dir),
        config["inference"],
        image_size=int(bundle["metadata"].get("image_size", config["train"]["stages"][-1]["image_size"])),
        normalization=str(config["model"].get("normalization", "half")),
        batch_size=args.batch_size,
        num_workers=suggested_num_workers(args.num_workers),
        device=device,
        tta=args.tta,
    )
    if list(filenames) != frame["filename"].tolist():
        raise RuntimeError("Evaluation ordering mismatch")
    print(json.dumps(metric_summary(frame[["lat", "lng"]].to_numpy(), prediction), indent=2))


if __name__ == "__main__":
    main()

