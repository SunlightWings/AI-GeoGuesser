from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

EARTH_RADIUS_KM = 6371.0088


def haversine_km_np(
    lat1: np.ndarray | float,
    lng1: np.ndarray | float,
    lat2: np.ndarray | float,
    lng2: np.ndarray | float,
) -> np.ndarray:
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    hav = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def pairwise_haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return haversine_km_np(a[:, 0, None], a[:, 1, None], b[None, :, 0], b[None, :, 1])


def latlng_to_unit_np(latlng: np.ndarray) -> np.ndarray:
    latlng = np.asarray(latlng, dtype=np.float64)
    lat = np.radians(latlng[:, 0])
    lng = np.radians(latlng[:, 1])
    return np.column_stack(
        (np.cos(lat) * np.cos(lng), np.cos(lat) * np.sin(lng), np.sin(lat))
    )


def unit_to_latlng_np(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12, None)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    lng = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return np.column_stack((lat, lng))


def latlng_to_unit_torch(latlng: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(latlng)
    lat, lng = radians[:, 0], radians[:, 1]
    return torch.stack(
        (torch.cos(lat) * torch.cos(lng), torch.cos(lat) * torch.sin(lng), torch.sin(lat)),
        dim=1,
    )


def unit_to_latlng_torch(xyz: torch.Tensor) -> torch.Tensor:
    xyz = F.normalize(xyz, dim=1)
    lat = torch.rad2deg(torch.asin(xyz[:, 2].clamp(-1.0, 1.0)))
    lng = torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0]))
    return torch.stack((lat, lng), dim=1)


def angular_distance_rad(pred_xyz: torch.Tensor, true_xyz: torch.Tensor) -> torch.Tensor:
    pred_xyz = F.normalize(pred_xyz, dim=-1)
    true_xyz = F.normalize(true_xyz, dim=-1)
    cross = torch.linalg.vector_norm(torch.cross(pred_xyz, true_xyz, dim=-1), dim=-1)
    dot = (pred_xyz * true_xyz).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.atan2(cross, dot)


def geodesic_medoid(latlng: np.ndarray) -> np.ndarray:
    """Return the observed point minimizing total great-circle distance."""
    latlng = np.asarray(latlng, dtype=np.float64)
    if len(latlng) == 0:
        raise ValueError("Cannot compute a medoid of an empty array")
    if len(latlng) == 1:
        return latlng[0].copy()
    distance = pairwise_haversine_km(latlng, latlng)
    return latlng[int(np.argmin(distance.sum(axis=1)))].copy()


def spherical_mean_np(latlng: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    xyz = latlng_to_unit_np(latlng)
    if weights is None:
        weights = np.ones(len(xyz), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)
    return unit_to_latlng_np((xyz * weights[:, None]).sum(axis=0, keepdims=True))[0]


def metric_summary(true_latlng: np.ndarray, pred_latlng: np.ndarray) -> dict[str, float]:
    true_latlng = np.asarray(true_latlng, dtype=np.float64)
    pred_latlng = np.asarray(pred_latlng, dtype=np.float64)
    if true_latlng.shape != pred_latlng.shape or true_latlng.ndim != 2 or true_latlng.shape[1] != 2:
        raise ValueError(f"Expected matching (N, 2) arrays; got {true_latlng.shape}, {pred_latlng.shape}")
    errors = haversine_km_np(
        true_latlng[:, 0], true_latlng[:, 1], pred_latlng[:, 0], pred_latlng[:, 1]
    )
    return {
        "median_km": float(np.median(errors)),
        "mean_km": float(np.mean(errors)),
        "lt_200_km": float(np.mean(errors < 200.0)),
        "lt_750_km": float(np.mean(errors < 750.0)),
        "p90_km": float(np.percentile(errors, 90.0)),
    }


def ensemble_latlng(predictions: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(item, dtype=np.float64) for item in predictions]
    if not arrays:
        raise ValueError("At least one prediction array is required")
    if any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("All prediction arrays must have the same shape")
    mean_xyz = np.mean([latlng_to_unit_np(item) for item in arrays], axis=0)
    return unit_to_latlng_np(mean_xyz)

