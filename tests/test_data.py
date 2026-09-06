import pandas as pd

from geoguesser.config import apply_overrides
from geoguesser.data import add_country_ids, make_folds


def test_random_folds_preserve_every_country():
    frame = pd.DataFrame(
        {
            "country": ["A"] * 10 + ["B"] * 10,
            "lat": list(range(10)) * 2,
            "lng": list(range(10)) * 2,
        }
    )
    frame, _ = add_country_ids(frame)
    folds = make_folds(frame, n_splits=5, seed=7, kind="random")
    assert set(folds) == set(range(5))
    for fold in range(5):
        assert set(frame.loc[folds == fold, "country"]) == {"A", "B"}


def test_nested_list_override():
    config = {
        "seed": 1,
        "data": {},
        "split": {"kind": "random", "n_folds": 2},
        "model": {"backbone": "efficientnet_b0", "embedding_dim": 1},
        "geocells": {},
        "loss": {},
        "train": {"stages": [{"image_size": 64, "epochs": 1, "batch_size": 2}]},
        "inference": {
            "top_cells": 1,
            "top_cells_train": 1,
            "temperature": 1.0,
            "residual_scale": 0.0,
            "global_blend": 0.0,
        },
        "rules": {"max_parameters": 5_000_000},
    }
    updated = apply_overrides(config, ["train.stages.0.epochs=3"])
    assert updated["train"]["stages"][0]["epochs"] == 3
