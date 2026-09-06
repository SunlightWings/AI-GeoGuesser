import numpy as np
import pandas as pd
import pytest

from geoguesser.leaderboard import (
    collect_oof_predictions,
    leaderboard_fold_assignments,
    read_leaderboard_folds,
    score_oof_predictions,
)


def sample_folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "country": ["A", "A", "B", "B"],
            "iso": ["AA", "AA", "BB", "BB"],
            "lat": [10.0, 11.0, 20.0, 21.0],
            "lng": [30.0, 31.0, 40.0, 41.0],
            "fold": [0, 1, 0, 1],
        }
    )


def test_shared_fold_assignments_follow_label_order(tmp_path):
    folds = sample_folds()
    path = tmp_path / "folds.csv"
    folds.sample(frac=1.0, random_state=2).to_csv(path, index=False)
    labels = folds.drop(columns="fold").iloc[[2, 0, 3, 1]].reset_index(drop=True)
    assignment = leaderboard_fold_assignments(labels, path, n_folds=2)
    assert assignment.tolist() == [0, 0, 1, 1]


def test_shared_folds_reject_changed_truth(tmp_path):
    folds = sample_folds()
    path = tmp_path / "folds.csv"
    folds.to_csv(path, index=False)
    labels = folds.drop(columns="fold")
    labels.loc[0, "lat"] += 1.0
    with pytest.raises(ValueError, match="lat values"):
        read_leaderboard_folds(path, labels=labels, n_folds=2)


def test_oof_collection_and_pooled_score(tmp_path):
    folds = sample_folds()
    for fold in (0, 1):
        subset = folds.loc[folds["fold"] == fold]
        path = tmp_path / f"fold_{fold}"
        path.mkdir()
        pd.DataFrame(
            {
                "filename": subset["filename"],
                "pred_lat": subset["lat"],
                "pred_lng": subset["lng"],
            }
        ).to_csv(path / "oof_predictions.csv", index=False)
    predictions = collect_oof_predictions(tmp_path, folds)
    report, _ = score_oof_predictions(folds, predictions)
    assert predictions["filename"].tolist() == folds["filename"].tolist()
    assert np.isclose(report["overall"]["median_km"], 0.0)
    assert np.isclose(report["overall"]["lt_200_km"], 1.0)


def test_oof_collection_rejects_wrong_fold_rows(tmp_path):
    folds = sample_folds()
    for fold in (0, 1):
        subset = folds.loc[folds["fold"] != fold]
        path = tmp_path / f"fold_{fold}"
        path.mkdir()
        pd.DataFrame(
            {
                "filename": subset["filename"],
                "pred_lat": subset["lat"],
                "pred_lng": subset["lng"],
            }
        ).to_csv(path / "oof_predictions.csv", index=False)
    with pytest.raises(ValueError, match="Prediction filenames"):
        collect_oof_predictions(tmp_path, folds)
