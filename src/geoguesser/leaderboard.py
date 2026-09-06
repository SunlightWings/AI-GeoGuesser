from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geo import metric_summary


FOLD_COLUMNS = ["filename", "country", "iso", "lat", "lng", "fold"]
PREDICTION_COLUMNS = ["filename", "pred_lat", "pred_lng"]


def read_leaderboard_folds(
    path: str | Path,
    labels: pd.DataFrame | None = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Read and strictly validate the shared unofficial-leaderboard folds."""
    frame = pd.read_csv(path)
    missing = set(FOLD_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Leaderboard folds are missing columns: {sorted(missing)}")
    if frame["filename"].duplicated().any():
        duplicates = frame.loc[frame["filename"].duplicated(), "filename"].head().tolist()
        raise ValueError(f"Leaderboard folds contain duplicate filenames: {duplicates}")
    if frame[FOLD_COLUMNS].isna().any().any():
        raise ValueError("Leaderboard folds contain missing values")

    numeric_fold = pd.to_numeric(frame["fold"], errors="coerce")
    if numeric_fold.isna().any() or not np.equal(numeric_fold, np.floor(numeric_fold)).all():
        raise ValueError("Leaderboard fold values must be integers")
    frame = frame.copy()
    frame["fold"] = numeric_fold.astype(np.int64)
    expected_folds = set(range(int(n_folds)))
    actual_folds = set(frame["fold"].unique())
    if actual_folds != expected_folds:
        raise ValueError(
            f"Leaderboard fold IDs must be {sorted(expected_folds)}, got {sorted(actual_folds)}"
        )
    if not frame["lat"].between(-90.0, 90.0).all():
        raise ValueError("Leaderboard folds contain latitude outside [-90, 90]")
    if not frame["lng"].between(-180.0, 180.0).all():
        raise ValueError("Leaderboard folds contain longitude outside [-180, 180]")

    if labels is not None:
        if labels["filename"].duplicated().any():
            raise ValueError("Training labels contain duplicate filenames")
        expected_names = set(labels["filename"])
        actual_names = set(frame["filename"])
        if actual_names != expected_names:
            missing_names = sorted(expected_names - actual_names)
            extra_names = sorted(actual_names - expected_names)
            raise ValueError(
                "Leaderboard folds do not match train_labels.csv. "
                f"Missing={missing_names[:5]}, extra={extra_names[:5]}"
            )
        joined = labels.set_index("filename").join(
            frame.set_index("filename"), how="left", rsuffix="_leaderboard", validate="one_to_one"
        )
        for column in ("country", "iso"):
            other = f"{column}_leaderboard"
            if not joined[column].astype(str).equals(joined[other].astype(str)):
                raise ValueError(f"Leaderboard {column} values do not match train_labels.csv")
        for column in ("lat", "lng"):
            other = f"{column}_leaderboard"
            if not np.allclose(
                joined[column].to_numpy(dtype=np.float64),
                joined[other].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(f"Leaderboard {column} values do not match train_labels.csv")
    return frame[FOLD_COLUMNS].reset_index(drop=True)


def leaderboard_fold_assignments(
    labels: pd.DataFrame, path: str | Path, n_folds: int = 5
) -> np.ndarray:
    """Return shared fold IDs in exactly the same row order as ``labels``."""
    folds = read_leaderboard_folds(path, labels=labels, n_folds=n_folds)
    lookup = folds.set_index("filename")["fold"]
    return labels["filename"].map(lookup).to_numpy(dtype=np.int64)


def validate_prediction_frame(
    predictions: pd.DataFrame, expected_names: set[str] | None = None
) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    result = predictions[PREDICTION_COLUMNS].copy()
    if result["filename"].duplicated().any():
        duplicates = result.loc[result["filename"].duplicated(), "filename"].head().tolist()
        raise ValueError(f"Predictions contain duplicate filenames: {duplicates}")
    values = result[["pred_lat", "pred_lng"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity")
    if not result["pred_lat"].between(-90.0, 90.0).all():
        raise ValueError("Predicted latitude outside [-90, 90]")
    if not result["pred_lng"].between(-180.0, 180.0).all():
        raise ValueError("Predicted longitude outside [-180, 180]")
    if expected_names is not None and set(result["filename"]) != expected_names:
        missing_names = sorted(expected_names - set(result["filename"]))
        extra_names = sorted(set(result["filename"]) - expected_names)
        raise ValueError(
            f"Prediction filenames do not match. Missing={missing_names[:5]}, extra={extra_names[:5]}"
        )
    return result


def collect_oof_predictions(run_dir: str | Path, folds: pd.DataFrame) -> pd.DataFrame:
    """Collect one held-out prediction file per fold and reject train-fold leakage."""
    run_dir = Path(run_dir)
    collected: list[pd.DataFrame] = []
    for fold in sorted(folds["fold"].unique()):
        expected_names = set(folds.loc[folds["fold"] == fold, "filename"])
        path = run_dir / f"fold_{fold}" / "oof_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing out-of-fold predictions: {path}")
        prediction = validate_prediction_frame(pd.read_csv(path), expected_names)
        prediction["fold"] = int(fold)
        collected.append(prediction)

    combined = pd.concat(collected, ignore_index=True)
    validate_prediction_frame(combined, set(folds["filename"]))
    expected_fold = folds.set_index("filename")["fold"]
    actual_fold = combined.set_index("filename")["fold"]
    if not actual_fold.sort_index().equals(expected_fold.sort_index()):
        raise ValueError("At least one prediction came from the wrong validation fold")
    order = folds[["filename"]].copy()
    return order.merge(
        combined[PREDICTION_COLUMNS], on="filename", how="left", validate="one_to_one"
    )


def score_oof_predictions(
    folds: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    prediction = validate_prediction_frame(predictions, set(folds["filename"]))
    merged = folds.merge(prediction, on="filename", how="left", validate="one_to_one")
    truth = merged[["lat", "lng"]].to_numpy(dtype=np.float64)
    guess = merged[["pred_lat", "pred_lng"]].to_numpy(dtype=np.float64)
    report: dict[str, Any] = {"overall": metric_summary(truth, guess), "per_fold": {}}
    for fold, subset in merged.groupby("fold", sort=True):
        report["per_fold"][str(int(fold))] = metric_summary(
            subset[["lat", "lng"]].to_numpy(dtype=np.float64),
            subset[["pred_lat", "pred_lng"]].to_numpy(dtype=np.float64),
        )
    return report, merged


def write_metrics(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
