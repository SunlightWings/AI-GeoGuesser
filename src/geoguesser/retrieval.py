from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import latlng_to_unit_np, unit_to_latlng_np


@dataclass(frozen=True)
class RetrievalConfig:
    top_cells: int = 5
    neighbors: int = 3
    embedding_temperature: float = 0.08
    cell_logprob_weight: float = 0.50
    retrieval_blend: float = 0.75


def _log_softmax(array: np.ndarray) -> np.ndarray:
    shifted = array - np.max(array, axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _softmax_1d(array: np.ndarray) -> np.ndarray:
    shifted = array - np.max(array)
    value = np.exp(shifted)
    return value / np.clip(value.sum(), 1e-12, None)


def hybrid_retrieval_predict(
    query_embedding: np.ndarray,
    query_fine_logits: np.ndarray,
    query_country_logits: np.ndarray,
    base_latlng: np.ndarray,
    bank_embedding: np.ndarray,
    bank_latlng: np.ndarray,
    bank_fine_cell: np.ndarray,
    cell_country_ids: np.ndarray,
    country_gate: float,
    config: RetrievalConfig,
) -> np.ndarray:
    """Refine neural coordinates with cell-gated train-only nearest neighbours."""
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    bank_embedding = np.asarray(bank_embedding, dtype=np.float32)
    query_embedding /= np.clip(np.linalg.norm(query_embedding, axis=1, keepdims=True), 1e-8, None)
    bank_embedding /= np.clip(np.linalg.norm(bank_embedding, axis=1, keepdims=True), 1e-8, None)
    bank_latlng = np.asarray(bank_latlng, dtype=np.float64)
    bank_fine_cell = np.asarray(bank_fine_cell, dtype=np.int64)
    cell_country_ids = np.asarray(cell_country_ids, dtype=np.int64)
    gated = _log_softmax(query_fine_logits) + country_gate * _log_softmax(query_country_logits)[:, cell_country_ids]
    indices_by_cell = [np.flatnonzero(bank_fine_cell == cell) for cell in range(len(cell_country_ids))]
    base_xyz = latlng_to_unit_np(base_latlng)
    result_xyz = np.empty_like(base_xyz)

    for index in range(len(query_embedding)):
        cell_count = min(max(1, config.top_cells), gated.shape[1])
        top_cells = np.argpartition(gated[index], -cell_count)[-cell_count:]
        candidate_parts = [indices_by_cell[int(cell)] for cell in top_cells if len(indices_by_cell[int(cell)])]
        if not candidate_parts:
            result_xyz[index] = base_xyz[index]
            continue
        candidate_indices = np.concatenate(candidate_parts)
        similarity = bank_embedding[candidate_indices] @ query_embedding[index]
        score = (
            similarity / config.embedding_temperature
            + config.cell_logprob_weight * gated[index, bank_fine_cell[candidate_indices]]
        )
        neighbors = min(max(1, config.neighbors), len(candidate_indices))
        selected_local = np.argpartition(score, -neighbors)[-neighbors:]
        selected = candidate_indices[selected_local]
        weight = _softmax_1d(score[selected_local])
        neighbor_xyz = (latlng_to_unit_np(bank_latlng[selected]) * weight[:, None]).sum(axis=0)
        neighbor_xyz /= np.clip(np.linalg.norm(neighbor_xyz), 1e-12, None)
        result_xyz[index] = (
            (1.0 - config.retrieval_blend) * base_xyz[index]
            + config.retrieval_blend * neighbor_xyz
        )
    return unit_to_latlng_np(result_xyz)

