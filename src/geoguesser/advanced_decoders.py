from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize_rows(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    return value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), eps, None)


def _log_softmax(array: np.ndarray, axis: int = 1) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    maximum = np.max(value, axis=axis, keepdims=True)
    shifted = value - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _softmax(array: np.ndarray, axis: int = 1) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    value = value - np.max(value, axis=axis, keepdims=True)
    value = np.exp(value)
    return value / np.clip(value.sum(axis=axis, keepdims=True), 1e-12, None)


def latlng_to_xyz(latlng: np.ndarray) -> np.ndarray:
    radians = np.radians(np.asarray(latlng, dtype=np.float64))
    latitude, longitude = radians[:, 0], radians[:, 1]
    return np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )


def xyz_to_latlng(xyz: np.ndarray) -> np.ndarray:
    value = _normalize_rows(xyz)
    latitude = np.degrees(np.arcsin(np.clip(value[:, 2], -1.0, 1.0)))
    longitude = np.degrees(np.arctan2(value[:, 1], value[:, 0]))
    return np.column_stack((latitude, longitude))


def _top_indices(score: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(max(1, int(count)), score.shape[1])
    if count == score.shape[1]:
        index = np.argsort(score, axis=1)[:, ::-1]
    else:
        index = np.argpartition(score, -count, axis=1)[:, -count:]
        selected = np.take_along_axis(score, index, axis=1)
        order = np.argsort(selected, axis=1)[:, ::-1]
        index = np.take_along_axis(index, order, axis=1)
    return index, np.take_along_axis(score, index, axis=1)


def decode_fine_numpy(
    output: dict[str, np.ndarray],
    centers_xyz: np.ndarray,
    cell_country_ids: np.ndarray,
    *,
    top_cells: int,
    temperature: float,
    country_gate: float,
    residual_scale: float,
    global_blend: float,
) -> np.ndarray:
    """NumPy equivalent of model.decode_outputs for cached-output sweeps."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    fine_logits = np.asarray(output["fine_logits"], dtype=np.float64)
    country_log_probability = _log_softmax(output["country_logits"])
    countries = np.asarray(cell_country_ids, dtype=np.int64)
    centers = _normalize_rows(centers_xyz)
    if fine_logits.shape[1] != len(centers) or len(countries) != len(centers):
        raise ValueError("Fine logits and fine-partition arrays disagree")

    score = fine_logits / float(temperature)
    score = score + float(country_gate) * country_log_probability[:, countries]
    top_index, top_score = _top_indices(score, top_cells)
    probability = _softmax(top_score)
    selected_centers = centers[top_index]
    residuals = np.asarray(output["fine_residuals"], dtype=np.float64)
    rows = np.arange(len(fine_logits))[:, None]
    selected_residuals = residuals[rows, top_index]
    candidate_xyz = _normalize_rows(
        selected_centers + float(residual_scale) * np.tanh(selected_residuals)
    )
    cell_xyz = _normalize_rows((probability[..., None] * candidate_xyz).sum(axis=1))
    global_xyz = _normalize_rows(output["global_xyz"])
    prediction_xyz = _normalize_rows(
        (1.0 - float(global_blend)) * cell_xyz + float(global_blend) * global_xyz
    )
    return xyz_to_latlng(prediction_xyz)


@dataclass(frozen=True)
class IntersectionPartition:
    fine_cell: np.ndarray
    coarse_cell: np.ndarray
    country_id: np.ndarray
    center_xyz: np.ndarray
    member_count: np.ndarray


def build_intersection_partition(
    latlng: np.ndarray,
    fine_cell: np.ndarray,
    coarse_cell: np.ndarray,
    country_id: np.ndarray,
) -> IntersectionPartition:
    """Build nonempty fine/coarse intersections from fold-training coordinates only."""
    coordinates = np.asarray(latlng, dtype=np.float64)
    fine = np.asarray(fine_cell, dtype=np.int64)
    coarse = np.asarray(coarse_cell, dtype=np.int64)
    country = np.asarray(country_id, dtype=np.int64)
    if not (len(coordinates) == len(fine) == len(coarse) == len(country)):
        raise ValueError("Intersection inputs must have equal row counts")
    key = np.column_stack((country, fine, coarse))
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    xyz = latlng_to_xyz(coordinates)
    centers = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.int64)
    np.add.at(centers, inverse, xyz)
    centers = _normalize_rows(centers)
    return IntersectionPartition(
        fine_cell=unique[:, 1].astype(np.int64),
        coarse_cell=unique[:, 2].astype(np.int64),
        country_id=unique[:, 0].astype(np.int64),
        center_xyz=centers,
        member_count=counts,
    )


def decode_cplanet_numpy(
    output: dict[str, np.ndarray],
    partition: IntersectionPartition,
    *,
    top_intersections: int,
    temperature: float,
    country_gate: float,
    coarse_weight: float,
) -> np.ndarray:
    """Product-of-experts decoder over fine/coarse geographic intersections."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    fine_logp = _log_softmax(
        np.asarray(output["fine_logits"], dtype=np.float64) / float(temperature)
    )
    coarse_logp = _log_softmax(
        np.asarray(output["coarse_logits"], dtype=np.float64) / float(temperature)
    )
    country_logp = _log_softmax(output["country_logits"])
    if int(partition.fine_cell.max()) >= fine_logp.shape[1]:
        raise ValueError("Intersection refers to an unknown fine cell")
    if int(partition.coarse_cell.max()) >= coarse_logp.shape[1]:
        raise ValueError("Intersection refers to an unknown coarse cell")
    if int(partition.country_id.max()) >= country_logp.shape[1]:
        raise ValueError("Intersection refers to an unknown country")

    score = fine_logp[:, partition.fine_cell]
    score += float(coarse_weight) * coarse_logp[:, partition.coarse_cell]
    score += float(country_gate) * country_logp[:, partition.country_id]
    top_index, top_score = _top_indices(score, top_intersections)
    probability = _softmax(top_score)
    prediction_xyz = _normalize_rows(
        (probability[..., None] * partition.center_xyz[top_index]).sum(axis=1)
    )
    return xyz_to_latlng(prediction_xyz)
