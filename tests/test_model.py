import numpy as np
import pandas as pd
import torch

from geoguesser.cells import CountryKMeansPartition
from geoguesser.model import build_model, decode_outputs, model_parameter_count


def partitions():
    frame = pd.DataFrame(
        {
            "country_id": [0] * 4 + [1] * 4,
            "lat": [50.0, 50.2, 51.0, 51.2, 60.0, 60.2, 61.0, 61.2],
            "lng": [10.0, 10.2, 11.0, 11.2, 20.0, 20.2, 21.0, 21.2],
        }
    )
    fine = CountryKMeansPartition(2, n_init=2)
    coarse = CountryKMeansPartition(4, n_init=2)
    fine.fit_transform(frame)
    coarse.fit_transform(frame)
    return fine, coarse


def test_mobilenet_forward_decode():
    fine, coarse = partitions()
    config = {
        "backbone": "mobilenet_v3_small",
        "pretrained": False,
        "embedding_dim": 32,
        "dropout": 0.0,
        "n_countries": 2,
        "n_fine_cells": fine.n_cells_,
        "n_coarse_cells": coarse.n_cells_,
    }
    model = build_model(config)
    output = model(torch.randn(2, 3, 64, 64))
    prediction, xyz = decode_outputs(output, fine, top_cells=2)
    assert prediction.shape == (2, 2)
    assert xyz.shape == (2, 3)
    assert torch.isfinite(prediction).all()
    assert model_parameter_count(model) < 5_000_000


def test_full_efficientnet_head_stays_under_cap():
    config = {
        "backbone": "efficientnet_b0",
        "pretrained": False,
        "embedding_dim": 256,
        "dropout": 0.15,
        "n_countries": 12,
        "n_fine_cells": 493,
        "n_coarse_cells": 119,
    }
    assert model_parameter_count(build_model(config)) == 4_876_618

