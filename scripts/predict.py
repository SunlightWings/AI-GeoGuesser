#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geoguesser.checkpoint import load_checkpoint
from geoguesser.data import suggested_num_workers
from geoguesser.geo import ensemble_latlng
from geoguesser.inference import predict_frame
from geoguesser.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate holdout predictions")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", choices=["none", "rot180", "rot4"], default=None)
    parser.add_argument("--top-cells", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--country-gate", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--global-blend", type=float, default=None)
    parser.add_argument("--expected-count", type=int, default=2400)
    parser.add_argument("--allow-noncompliant-checkpoint", action="store_true")
    parser.add_argument("--allow-oversized-ensemble", action="store_true")
    return parser.parse_args()


def validate_predictions(frame: pd.DataFrame, expected_names: set[str], expected_count: int) -> None:
    if list(frame.columns) != ["filename", "pred_lat", "pred_lng"]:
        raise ValueError("Submission header must be exactly filename,pred_lat,pred_lng")
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} predictions, found {len(frame)}")
    if frame["filename"].duplicated().any():
        raise ValueError("Submission filenames must be unique")
    if set(frame["filename"]) != expected_names:
        missing = sorted(expected_names - set(frame["filename"]))
        extra = sorted(set(frame["filename"]) - expected_names)
        raise ValueError(f"Filename mismatch. Missing={missing[:5]}, extra={extra[:5]}")
    values = frame[["pred_lat", "pred_lng"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity")
    if not frame["pred_lat"].between(-90.0, 90.0).all():
        raise ValueError("Predicted latitude outside [-90, 90]")
    if not frame["pred_lng"].between(-180.0, 180.0).all():
        raise ValueError("Predicted longitude outside [-180, 180]")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    workers = suggested_num_workers(args.num_workers)
    holdout_dir = Path(args.data_root).resolve() / "holdout_public"
    names = sorted(path.name for path in holdout_dir.glob("*.jpg"))
    if len(names) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} holdout images, found {len(names)}")
    frame = pd.DataFrame({"filename": names})

    all_predictions: list[np.ndarray] = []
    total_parameters = 0
    for checkpoint_path in args.checkpoint:
        model, fine_partition, _, bundle = load_checkpoint(checkpoint_path, device)
        if not bundle["rule_compliant"] and not args.allow_noncompliant_checkpoint:
            raise ValueError(
                f"{checkpoint_path} used external pretrained weights. Pass "
                "--allow-noncompliant-checkpoint only for diagnostics, never silent submission."
            )
        total_parameters += int(bundle["parameter_count"])
        config = bundle["config"]
        inference_config = dict(config["inference"])
        for cli_name, config_name in (
            ("top_cells", "top_cells"),
            ("temperature", "temperature"),
            ("country_gate", "country_gate"),
            ("residual_scale", "residual_scale"),
            ("global_blend", "global_blend"),
        ):
            value = getattr(args, cli_name)
            if value is not None:
                inference_config[config_name] = value
        tta = args.tta or str(config["inference"].get("tta", "none"))
        image_size = int(bundle["metadata"].get("image_size", config["train"]["stages"][-1]["image_size"]))
        filenames, prediction = predict_frame(
            model,
            fine_partition,
            frame,
            holdout_dir,
            inference_config,
            image_size=image_size,
            normalization=str(config["model"].get("normalization", "half")),
            batch_size=args.batch_size,
            num_workers=workers,
            device=device,
            tta=tta,
        )
        if list(filenames) != names:
            raise RuntimeError("Inference changed holdout ordering")
        all_predictions.append(prediction)
    if len(all_predictions) > 1 and total_parameters > 5_000_000 and not args.allow_oversized_ensemble:
        raise ValueError(
            f"Ensemble totals {total_parameters:,} parameters and may violate the cap. "
            "Use one final model or obtain written approval."
        )
    prediction = ensemble_latlng(all_predictions)
    output = pd.DataFrame(
        {"filename": names, "pred_lat": prediction[:, 0], "pred_lng": prediction[:, 1]}
    )
    validate_predictions(output, set(names), args.expected_count)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8", float_format="%.8f")
    print(json.dumps({"output": str(output_path), "rows": len(output), "models": len(all_predictions)}))


if __name__ == "__main__":
    main()
