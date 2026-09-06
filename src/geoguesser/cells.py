from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .geo import geodesic_medoid, latlng_to_unit_np, unit_to_latlng_np


@dataclass
class CountryKMeansPartition:
    """Balanced spherical geocells fitted separately within every country."""

    target_size: int
    random_state: int = 2026
    n_init: int = 20

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        required = {"country_id", "lat", "lng"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Partition input is missing columns: {sorted(missing)}")
        labels_out = np.full(len(frame), -1, dtype=np.int64)
        decision_centers: list[np.ndarray] = []
        output_centers: list[np.ndarray] = []
        cell_country_ids: list[int] = []
        offset = 0
        country_values = frame["country_id"].to_numpy(dtype=np.int64)

        for country_id in sorted(np.unique(country_values)):
            positions = np.flatnonzero(country_values == country_id)
            latlng = frame.iloc[positions][["lat", "lng"]].to_numpy(dtype=np.float64)
            xyz = latlng_to_unit_np(latlng)
            n_cells = min(len(positions), max(1, round(len(positions) / self.target_size)))
            clustering = KMeans(
                n_clusters=n_cells,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            local_labels = clustering.fit_predict(xyz)
            centers_xyz = clustering.cluster_centers_
            centers_xyz /= np.clip(np.linalg.norm(centers_xyz, axis=1, keepdims=True), 1e-12, None)

            for local_cell in range(n_cells):
                members = latlng[local_labels == local_cell]
                decision_centers.append(centers_xyz[local_cell])
                output_centers.append(geodesic_medoid(members))
                cell_country_ids.append(int(country_id))

            labels_out[positions] = local_labels + offset
            offset += n_cells

        if np.any(labels_out < 0):
            raise RuntimeError("Not every row received a geocell")
        self.decision_centers_xyz_ = np.asarray(decision_centers, dtype=np.float32)
        self.decision_centers_latlng_ = unit_to_latlng_np(self.decision_centers_xyz_).astype(np.float32)
        self.output_centers_latlng_ = np.asarray(output_centers, dtype=np.float32)
        self.output_centers_xyz_ = latlng_to_unit_np(self.output_centers_latlng_).astype(np.float32)
        self.cell_country_ids_ = np.asarray(cell_country_ids, dtype=np.int64)
        self.n_cells_ = int(len(self.cell_country_ids_))
        return labels_out

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        country_values = frame["country_id"].to_numpy(dtype=np.int64)
        query_xyz = latlng_to_unit_np(frame[["lat", "lng"]].to_numpy(dtype=np.float64))
        result = np.full(len(frame), -1, dtype=np.int64)
        for country_id in sorted(np.unique(country_values)):
            rows = np.flatnonzero(country_values == country_id)
            candidates = np.flatnonzero(self.cell_country_ids_ == country_id)
            if not len(candidates):
                raise ValueError(f"No fitted cells exist for country id {country_id}")
            similarity = query_xyz[rows] @ self.decision_centers_xyz_[candidates].T
            result[rows] = candidates[np.argmax(similarity, axis=1)]
        return result

    def to_state(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "target_size": self.target_size,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "decision_centers_xyz": self.decision_centers_xyz_,
            "decision_centers_latlng": self.decision_centers_latlng_,
            "output_centers_latlng": self.output_centers_latlng_,
            "output_centers_xyz": self.output_centers_xyz_,
            "cell_country_ids": self.cell_country_ids_,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CountryKMeansPartition":
        result = cls(
            target_size=int(state["target_size"]),
            random_state=int(state["random_state"]),
            n_init=int(state.get("n_init", 20)),
        )
        for name in (
            "decision_centers_xyz",
            "decision_centers_latlng",
            "output_centers_latlng",
            "output_centers_xyz",
            "cell_country_ids",
        ):
            dtype = np.int64 if name == "cell_country_ids" else np.float32
            setattr(result, f"{name}_", np.asarray(state[name], dtype=dtype))
        result.n_cells_ = int(len(result.cell_country_ids_))
        return result

    def _check_fitted(self) -> None:
        if not hasattr(self, "n_cells_"):
            raise RuntimeError("Partition has not been fitted")
