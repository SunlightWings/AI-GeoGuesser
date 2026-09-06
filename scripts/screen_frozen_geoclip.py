#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geoguesser.frozen_geoclip import (
    TinyGpsTower,
    haversine_rbf_features,
    metric_summary,
    multiscale_rff_features,
    nearest_candidate,
    normalize_rows,
    spherical_xyz_features,
    train_tower,
)


def load_validation_cache(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"filename", "truth_latlng", "embedding"}
        if missing := required - set(archive.files):
            raise ValueError(f"Validation cache is missing: {sorted(missing)}")
        metadata = json.loads(str(archive["metadata"].item()))
        return archive["filename"], archive["truth_latlng"], archive["embedding"], metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tiny GPS encoders while keeping the image network frozen"
    )
    parser.add_argument("--reference-bank", required=True, type=Path)
    parser.add_argument("--folds-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-cache", default=None, type=Path)
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument(
        "--methods", nargs="+", choices=["xyz", "rff", "rbf"], default=["xyz", "rff", "rbf"]
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(args.folds_csv)
    if not {"filename", "lat", "lng", "fold"}.issubset(folds.columns):
        raise ValueError("folds.csv must contain filename, lat, lng, fold")
    fold_lookup = folds.set_index("filename")
    with np.load(args.reference_bank, allow_pickle=False) as bank:
        bank_names = bank["filename"].astype(str)
        bank_embedding = normalize_rows(bank["embedding"].astype(np.float32))
        bank_latlng = bank["latlng"].astype(np.float32)
        bank_metadata = json.loads(str(bank["metadata"].item()))
    if len(set(bank_names)) != len(bank_names):
        raise ValueError("Reference bank has duplicate filenames")
    missing = sorted(set(bank_names) - set(fold_lookup.index))
    if missing:
        raise ValueError(f"Reference-bank names absent from folds.csv: {missing[:5]}")
    expected_latlng = fold_lookup.loc[bank_names, ["lat", "lng"]].to_numpy(dtype=np.float32)
    if not np.allclose(bank_latlng, expected_latlng, rtol=0.0, atol=1e-5):
        raise ValueError("Reference-bank coordinates disagree with folds.csv")

    if args.validation_cache is None:
        bank_folds = fold_lookup.loc[bank_names, "fold"].to_numpy(dtype=np.int64)
        validation_mask = bank_folds == args.validation_fold
        training_mask = ~validation_mask
        if not validation_mask.any() or not training_mask.any():
            raise ValueError("Requested GPS-tower split is absent from the reference bank")
        train_names = bank_names[training_mask]
        train_embedding = bank_embedding[training_mask]
        train_latlng = bank_latlng[training_mask]
        validation_names = bank_names[validation_mask]
        validation_embedding = bank_embedding[validation_mask]
        validation_latlng = bank_latlng[validation_mask]
        scope = "GPS-tower-held-out only: image encoder previously trained on both subsets"
    else:
        train_names = bank_names
        train_embedding = bank_embedding
        train_latlng = bank_latlng
        validation_names, validation_latlng, validation_embedding, validation_metadata = (
            load_validation_cache(args.validation_cache)
        )
        validation_names = validation_names.astype(str)
        validation_latlng = validation_latlng.astype(np.float32)
        validation_embedding = normalize_rows(validation_embedding.astype(np.float32))
        if set(train_names) & set(validation_names):
            raise ValueError("Training bank and validation cache overlap")
        validation_missing = sorted(set(validation_names) - set(fold_lookup.index))
        if validation_missing:
            raise ValueError(
                f"Validation-cache names absent from folds.csv: {validation_missing[:5]}"
            )
        validation_truth = fold_lookup.loc[
            validation_names, ["lat", "lng"]
        ].to_numpy(dtype=np.float32)
        if not np.allclose(validation_latlng, validation_truth, rtol=0.0, atol=1e-5):
            raise ValueError("Validation-cache coordinates disagree with folds.csv")
        bank_hash = bank_metadata.get("checkpoint_sha256")
        validation_hash = validation_metadata.get("checkpoint_sha256")
        if bank_hash and validation_hash and bank_hash != validation_hash:
            raise ValueError("Reference bank and validation cache use different checkpoints")
        scope = "honest fold: frozen image encoder did not train on validation images"

    print(json.dumps({"training_rows": len(train_names), "validation_rows": len(validation_names), "scope": scope}, indent=2))
    results: list[dict[str, object]] = []
    image_knn_prediction = nearest_candidate(validation_embedding, train_embedding, train_latlng)
    results.append(
        {
            "method": "image_knn_control",
            "tower_parameters": 0,
            "final_train_loss": np.nan,
            "description": "nearest frozen training-image embedding",
            "scope": scope,
            **metric_summary(validation_latlng, image_knn_prediction),
        }
    )

    for method in args.methods:
        if method == "xyz":
            encoding = spherical_xyz_features(train_latlng, validation_latlng)
        elif method == "rff":
            encoding = multiscale_rff_features(
                train_latlng, validation_latlng, seed=args.seed, feature_count=64
            )
        else:
            encoding = haversine_rbf_features(
                train_latlng, validation_latlng, seed=args.seed, center_count=64
            )
        tower = TinyGpsTower(
            input_dim=encoding.train.shape[1],
            output_dim=train_embedding.shape[1],
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
        history = train_tower(
            tower,
            encoding.train,
            train_embedding,
            epochs=args.epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        _, _, candidate_embedding = tower.forward(encoding.train)
        prediction = nearest_candidate(validation_embedding, candidate_embedding, train_latlng)
        row = {
            "method": method,
            "tower_parameters": tower.parameter_count,
            "final_train_loss": history[-1],
            "description": encoding.description,
            "scope": scope,
            **metric_summary(validation_latlng, prediction),
        }
        results.append(row)
        pd.DataFrame(
            {
                "filename": validation_names,
                "true_lat": validation_latlng[:, 0],
                "true_lng": validation_latlng[:, 1],
                "pred_lat": prediction[:, 0],
                "pred_lng": prediction[:, 1],
            }
        ).to_csv(args.output_dir / f"predictions_{method}.csv", index=False)
        (args.output_dir / f"history_{method}.json").write_text(
            json.dumps({"loss": history}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row, indent=2))

    result = pd.DataFrame(results).sort_values(["median_km", "mean_km"])
    result.to_csv(args.output_dir / "results.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "reference_bank": str(args.reference_bank.resolve()),
                "folds_csv": str(args.folds_csv.resolve()),
                "validation_cache": None if args.validation_cache is None else str(args.validation_cache.resolve()),
                "validation_fold": args.validation_fold,
                "training_rows": len(train_names),
                "validation_rows": len(validation_names),
                "scope": scope,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "temperature": args.temperature,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("\nFrozen GeoCLIP screen:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
