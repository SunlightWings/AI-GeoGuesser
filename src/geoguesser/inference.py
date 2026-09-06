from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .cells import CountryKMeansPartition
from .data import GeoDataset, build_transforms
from .model import CompactGeoModel, decode_outputs


def _aggregate_outputs(outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in outputs[0]:
        value = torch.stack([item[key] for item in outputs]).mean(dim=0)
        if key in {"embedding", "global_xyz"}:
            value = F.normalize(value, dim=-1)
        result[key] = value
    return result


def forward_tta(
    model: CompactGeoModel, images: torch.Tensor, mode: str
) -> dict[str, torch.Tensor]:
    if mode == "none":
        return model(images)
    if mode == "rot180":
        variants = [images, torch.rot90(images, 2, dims=(-2, -1))]
    elif mode == "rot4":
        variants = [torch.rot90(images, k, dims=(-2, -1)) for k in range(4)]
    else:
        raise ValueError("TTA mode must be none, rot180, or rot4")
    return _aggregate_outputs([model(variant) for variant in variants])


@torch.inference_mode()
def extract_outputs(
    model: CompactGeoModel,
    frame,
    image_dir: str | Path,
    image_size: int,
    normalization: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    tta: str = "none",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dataset = GeoDataset(
        frame,
        image_dir,
        build_transforms(image_size, train=False, normalization=normalization),
        with_targets=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    filenames: list[str] = []
    collected: dict[str, list[np.ndarray]] = {}
    for batch in tqdm(loader, desc="extract", leave=False):
        output = forward_tta(model, batch["image"].to(device, non_blocking=True), tta)
        filenames.extend(batch["filename"])
        for key, value in output.items():
            collected.setdefault(key, []).append(value.float().cpu().numpy())
    return np.asarray(filenames), {key: np.concatenate(value) for key, value in collected.items()}


@torch.inference_mode()
def predict_frame(
    model: CompactGeoModel,
    partition: CountryKMeansPartition,
    frame,
    image_dir: str | Path,
    inference_config: dict[str, Any],
    image_size: int,
    normalization: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    tta: str,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = GeoDataset(
        frame,
        image_dir,
        build_transforms(image_size, train=False, normalization=normalization),
        with_targets=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    filenames: list[str] = []
    predictions: list[np.ndarray] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        output = forward_tta(model, images, tta)
        latlng, _ = decode_outputs(
            output,
            partition,
            top_cells=int(inference_config["top_cells"]),
            temperature=float(inference_config["temperature"]),
            country_gate=float(inference_config["country_gate"]),
            residual_scale=float(inference_config["residual_scale"]),
            global_blend=float(inference_config["global_blend"]),
        )
        filenames.extend(batch["filename"])
        predictions.append(latlng.cpu().numpy())
    return np.asarray(filenames), np.concatenate(predictions)
