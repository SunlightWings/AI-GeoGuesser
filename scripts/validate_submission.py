#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate(path: Path, holdout_dir: Path, expected_count: int) -> None:
    frame = pd.read_csv(path)
    expected_columns = ["filename", "pred_lat", "pred_lng"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"Columns must be exactly {expected_columns}, got {frame.columns.tolist()}")
    expected_names = {item.name for item in holdout_dir.glob("*.jpg")}
    if len(expected_names) != expected_count:
        raise ValueError(f"Holdout directory contains {len(expected_names)}, expected {expected_count}")
    if len(frame) != expected_count or frame["filename"].nunique() != expected_count:
        raise ValueError("Submission must contain exactly one row per holdout image")
    if set(frame["filename"]) != expected_names:
        raise ValueError("Submission filenames do not exactly match holdout_public")
    numeric = frame[["pred_lat", "pred_lng"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Coordinates contain NaN or infinity")
    if not frame["pred_lat"].between(-90, 90).all():
        raise ValueError("Latitude outside [-90, 90]")
    if not frame["pred_lng"].between(-180, 180).all():
        raise ValueError("Longitude outside [-180, 180]")
    print(f"PASS: {path} contains {expected_count} valid predictions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--holdout-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=2400)
    args = parser.parse_args()
    validate(args.predictions, args.holdout_dir, args.expected_count)


if __name__ == "__main__":
    main()
