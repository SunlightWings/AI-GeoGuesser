#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geoguesser.checkpoint import load_checkpoint
from geoguesser.data import suggested_num_workers
from geoguesser.inference import extract_outputs
from geoguesser.model import decode_outputs
from geoguesser.retrieval import RetrievalConfig, hybrid_retrieval_predict
from geoguesser.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with an optional train-only retrieval bank")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-bank", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", choices=["none", "rot180", "rot4"], default="rot180")
    parser.add_argument("--top-cells", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--embedding-temperature", type=float, default=0.08)
    parser.add_argument("--cell-logprob-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-blend", type=float, default=0.75)
    parser.add_argument("--expected-count", type=int, default=2400)
    parser.add_argument("--acknowledge-reference-bank-rule", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_reference_bank_rule:
        raise SystemExit("Obtain instructor clarification, then acknowledge the reference-bank rule.")
    device = choose_device(args.device)
    model, fine, _, bundle = load_checkpoint(args.checkpoint, device)
    if not bundle["rule_compliant"]:
        raise ValueError("External-pretrained checkpoints are not accepted by this submission path")
    names = sorted(path.name for path in (args.data_root / "holdout_public").glob("*.jpg"))
    if len(names) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} holdout images, found {len(names)}")
    frame = pd.DataFrame({"filename": names})
    config = bundle["config"]
    filenames, output = extract_outputs(
        model,
        frame,
        args.data_root / "holdout_public",
        image_size=int(bundle["metadata"]["image_size"]),
        normalization=str(config["model"].get("normalization", "half")),
        batch_size=args.batch_size,
        num_workers=suggested_num_workers(args.num_workers),
        device=device,
        tta=args.tta,
    )
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
    with np.load(args.reference_bank, allow_pickle=False) as bank:
        prediction = hybrid_retrieval_predict(
            output["embedding"],
            output["fine_logits"],
            output["country_logits"],
            base.cpu().numpy(),
            bank["embedding"],
            bank["latlng"],
            bank["fine_cell"],
            fine.cell_country_ids_,
            country_gate=float(config["inference"]["country_gate"]),
            config=RetrievalConfig(
                args.top_cells,
                args.neighbors,
                args.embedding_temperature,
                args.cell_logprob_weight,
                args.retrieval_blend,
            ),
        )
    result = pd.DataFrame(
        {"filename": filenames, "pred_lat": prediction[:, 0], "pred_lng": prediction[:, 1]}
    )
    if result["filename"].tolist() != names or not np.isfinite(prediction).all():
        raise RuntimeError("Retrieval prediction validation failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.8f")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

