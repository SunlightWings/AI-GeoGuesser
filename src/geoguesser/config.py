from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and retain its source path."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply command-line overrides such as ``train.epochs=2``."""
    result = copy.deepcopy(config)
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must have KEY=VALUE form: {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if isinstance(cursor, list):
                cursor = cursor[int(part)]
            else:
                if part not in cursor:
                    cursor[part] = {}
                cursor = cursor[part]
        final = parts[-1]
        if isinstance(cursor, list):
            cursor[int(final)] = value
        else:
            cursor[final] = value
    validate_config(result)
    return result


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "seed",
        "data",
        "split",
        "model",
        "geocells",
        "loss",
        "train",
        "inference",
        "rules",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    if config["model"]["backbone"] not in {"efficientnet_b0", "mobilenet_v3_small"}:
        raise ValueError("model.backbone must be efficientnet_b0 or mobilenet_v3_small")
    if int(config["model"]["embedding_dim"]) <= 0:
        raise ValueError("model.embedding_dim must be positive")
    if int(config["rules"]["max_parameters"]) > 5_000_000:
        raise ValueError("The challenge cap is 5,000,000 parameters")
    split_kind = str(config.get("split", {}).get("kind", "random"))
    if split_kind not in {"random", "spatial", "predefined"}:
        raise ValueError("split.kind must be random, spatial, or predefined")
    if split_kind == "predefined" and not config["split"].get("folds_csv"):
        raise ValueError("split.kind=predefined requires split.folds_csv")
    if int(config["split"].get("n_folds", 0)) < 2:
        raise ValueError("split.n_folds must be at least 2")
    rotation_probability = float(config["data"].get("rotation_probability", 0.0))
    if not 0.0 <= rotation_probability <= 1.0:
        raise ValueError("data.rotation_probability must be in [0, 1]")
    ema_decay = float(config["train"].get("ema_decay", 0.0))
    if ema_decay != 0.0:
        raise ValueError(
            "train.ema_decay must be 0. EMA was disabled because averaged scratch-CNN weights "
            "produced invalid BatchNorm inference statistics."
        )
    stages = config["train"].get("stages", [])
    if not stages:
        raise ValueError("train.stages must contain at least one stage")
    for stage in stages:
        if min(int(stage["image_size"]), int(stage["epochs"]), int(stage["batch_size"])) <= 0:
            raise ValueError(f"Invalid training stage: {stage}")
    inference = config["inference"]
    if int(inference["top_cells"]) <= 0 or int(inference["top_cells_train"]) <= 0:
        raise ValueError("inference top-cell counts must be positive")
    if float(inference["temperature"]) <= 0.0:
        raise ValueError("inference.temperature must be positive")
    if float(inference["residual_scale"]) < 0.0:
        raise ValueError("inference.residual_scale must be non-negative")
    if not 0.0 <= float(inference["global_blend"]) <= 1.0:
        raise ValueError("inference.global_blend must be in [0, 1]")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable copy without private runtime keys."""
    return {key: value for key, value in copy.deepcopy(config).items() if not key.startswith("_")}
