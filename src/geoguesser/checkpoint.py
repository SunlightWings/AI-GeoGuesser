from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .cells import CountryKMeansPartition
from .config import public_config
from .model import CompactGeoModel, build_model, model_parameter_count
from .utils import atomic_torch_save


def make_checkpoint_bundle(
    model: CompactGeoModel,
    model_config: dict[str, Any],
    fine_partition: CountryKMeansPartition,
    coarse_partition: CountryKMeansPartition,
    country_to_id: dict[str, int],
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    external_pretraining = bool(model_config.get("pretrained", False))
    return {
        "format_version": 1,
        "model_state": model.state_dict(),
        "model_config": dict(model_config),
        "fine_partition": fine_partition.to_state(),
        "coarse_partition": coarse_partition.to_state(),
        "country_to_id": dict(country_to_id),
        "config": public_config(config),
        "parameter_count": model_parameter_count(model),
        "rule_compliant": not external_pretraining,
        "external_pretraining": external_pretraining,
        "metadata": dict(metadata),
    }


def save_checkpoint(path: str | Path, **kwargs: Any) -> None:
    atomic_torch_save(make_checkpoint_bundle(**kwargs), path)


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[CompactGeoModel, CountryKMeansPartition, CountryKMeansPartition, dict[str, Any]]:
    bundle = torch.load(path, map_location=device, weights_only=False)
    if bundle.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    # The state dict already contains all learned weights. Never trigger an external download here.
    model = build_model(bundle["model_config"], load_external_weights=False)
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(device)
    fine = CountryKMeansPartition.from_state(bundle["fine_partition"])
    coarse = CountryKMeansPartition.from_state(bundle["coarse_partition"])
    return model, fine, coarse, bundle

