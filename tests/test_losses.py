import pandas as pd
import torch

from geoguesser.cells import CountryKMeansPartition
from geoguesser.losses import GeoCriterion
from geoguesser.model import build_model


def test_loss_is_finite_and_differentiable():
    frame = pd.DataFrame(
        {
            "country_id": [0, 0, 1, 1],
            "lat": [50.0, 51.0, 60.0, 61.0],
            "lng": [10.0, 11.0, 20.0, 21.0],
        }
    )
    fine = CountryKMeansPartition(1, n_init=2)
    coarse = CountryKMeansPartition(2, n_init=2)
    fine_labels = fine.fit_transform(frame)
    coarse_labels = coarse.fit_transform(frame)
    model = build_model(
        {
            "backbone": "mobilenet_v3_small",
            "pretrained": False,
            "embedding_dim": 32,
            "dropout": 0.0,
            "n_countries": 2,
            "n_fine_cells": fine.n_cells_,
            "n_coarse_cells": coarse.n_cells_,
        }
    )
    config = {
        "loss": {
            "fine_tau_km": 70.0,
            "coarse_tau_km": 180.0,
            "soft_target_fraction": 0.2,
            "country_label_smoothing": 0.0,
            "contrastive_temperature": 0.1,
            "weights": {"fine": 1.0, "coarse": 0.3, "country": 0.3, "global": 0.2, "residual": 0.5, "decoded": 1.0, "contrastive": 0.1},
        },
        "inference": {"top_cells_train": 2, "temperature": 1.0, "country_gate": 0.5, "residual_scale": 0.1, "global_blend": 0.1},
    }
    criterion = GeoCriterion(fine, coarse, config)
    latlng = torch.tensor(frame[["lat", "lng"]].to_numpy(), dtype=torch.float32)
    radians = torch.deg2rad(latlng)
    xyz = torch.stack((torch.cos(radians[:, 0]) * torch.cos(radians[:, 1]), torch.cos(radians[:, 0]) * torch.sin(radians[:, 1]), torch.sin(radians[:, 0])), dim=1)
    batch = {
        "latlng": latlng,
        "xyz": xyz,
        "country_id": torch.tensor(frame["country_id"].to_numpy()),
        "fine_cell": torch.tensor(fine_labels),
        "coarse_cell": torch.tensor(coarse_labels),
    }
    output = model(torch.randn(4, 3, 64, 64))
    loss, _ = criterion(output, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
