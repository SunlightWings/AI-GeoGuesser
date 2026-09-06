#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny dataset from attached sample images")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train_dir = args.output / "train"
    holdout_dir = args.output / "holdout_public"
    train_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels)
    available = sorted(args.images.glob("*.jpg"))
    labelled_names = set(labels["filename"])
    train_paths = [path for path in available if path.name in labelled_names]
    holdout_paths = [path for path in available if path.name not in labelled_names]
    smoke_labels = labels[labels["filename"].isin([path.name for path in train_paths])].copy()
    if len(train_paths) < 4 or not holdout_paths:
        raise ValueError("Need at least four labelled samples and one unlabelled sample")
    for path in train_paths:
        shutil.copy2(path, train_dir / path.name)
    for path in holdout_paths:
        shutil.copy2(path, holdout_dir / path.name)
    smoke_labels.to_csv(args.output / "train_labels.csv", index=False)
    print(f"Created smoke dataset: {len(train_paths)} train, {len(holdout_paths)} holdout")


if __name__ == "__main__":
    main()

