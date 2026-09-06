from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import CountryKMeansPartition
from .geo import EARTH_RADIUS_KM, angular_distance_rad, latlng_to_unit_torch
from .model import decode_outputs


def pairwise_haversine_torch(latlng: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    query_xyz = latlng_to_unit_torch(latlng)
    center_xyz = latlng_to_unit_torch(centers)
    dot = (query_xyz @ center_xyz.T).clamp(-1.0, 1.0)
    return EARTH_RADIUS_KM * torch.acos(dot)


def haversine_soft_targets(
    latlng: torch.Tensor,
    hard_cell: torch.Tensor,
    country_id: torch.Tensor,
    centers_latlng: torch.Tensor,
    cell_country_ids: torch.Tensor,
    tau_km: float,
    soft_fraction: float,
) -> torch.Tensor:
    distance = pairwise_haversine_torch(latlng, centers_latlng)
    allowed = country_id[:, None] == cell_country_ids[None, :]
    logits = (-distance / tau_km).masked_fill(~allowed, -1e4)
    spatial = F.softmax(logits, dim=1)
    one_hot = F.one_hot(hard_cell, num_classes=centers_latlng.shape[0]).float()
    return (1.0 - soft_fraction) * one_hot + soft_fraction * spatial


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def supervised_contrastive_loss(
    embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10
) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    embedding = F.normalize(embedding, dim=1)
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (labels[:, None] == labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not valid.any():
        return embedding.sum() * 0.0
    similarity = similarity.masked_fill(identity, -1e4)
    log_probability = F.log_softmax(similarity, dim=1)
    positives_per_row = positive.sum(dim=1).clamp_min(1)
    per_anchor = -(log_probability * positive).sum(dim=1) / positives_per_row
    return per_anchor[valid].mean()


class GeoCriterion(nn.Module):
    def __init__(
        self,
        fine_partition: CountryKMeansPartition,
        coarse_partition: CountryKMeansPartition,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.fine_partition = fine_partition
        self.config = config
        self.register_buffer(
            "fine_centers_latlng",
            torch.as_tensor(fine_partition.decision_centers_latlng_, dtype=torch.float32),
        )
        self.register_buffer(
            "fine_centers_xyz",
            torch.as_tensor(fine_partition.output_centers_xyz_, dtype=torch.float32),
        )
        self.register_buffer(
            "fine_cell_countries",
            torch.as_tensor(fine_partition.cell_country_ids_, dtype=torch.long),
        )
        self.register_buffer(
            "coarse_centers_latlng",
            torch.as_tensor(coarse_partition.decision_centers_latlng_, dtype=torch.float32),
        )
        self.register_buffer(
            "coarse_cell_countries",
            torch.as_tensor(coarse_partition.cell_country_ids_, dtype=torch.long),
        )

    def forward(
        self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_config = self.config["loss"]
        inference_config = self.config["inference"]
        fine_target = haversine_soft_targets(
            batch["latlng"],
            batch["fine_cell"],
            batch["country_id"],
            self.fine_centers_latlng,
            self.fine_cell_countries,
            tau_km=float(loss_config["fine_tau_km"]),
            soft_fraction=float(loss_config["soft_target_fraction"]),
        )
        coarse_target = haversine_soft_targets(
            batch["latlng"],
            batch["coarse_cell"],
            batch["country_id"],
            self.coarse_centers_latlng,
            self.coarse_cell_countries,
            tau_km=float(loss_config["coarse_tau_km"]),
            soft_fraction=float(loss_config["soft_target_fraction"]),
        )
        fine_loss = soft_cross_entropy(output["fine_logits"], fine_target)
        coarse_loss = soft_cross_entropy(output["coarse_logits"], coarse_target)
        country_loss = F.cross_entropy(
            output["country_logits"],
            batch["country_id"],
            label_smoothing=float(loss_config["country_label_smoothing"]),
        )
        global_loss = angular_distance_rad(output["global_xyz"], batch["xyz"]).mean()

        row = torch.arange(len(batch["fine_cell"]), device=batch["fine_cell"].device)
        residual = output["fine_residuals"][row, batch["fine_cell"]]
        local_xyz = F.normalize(
            self.fine_centers_xyz[batch["fine_cell"]]
            + float(inference_config["residual_scale"]) * torch.tanh(residual),
            dim=1,
        )
        residual_loss = angular_distance_rad(local_xyz, batch["xyz"]).mean()
        _, decoded_xyz = decode_outputs(
            output,
            self.fine_partition,
            top_cells=int(inference_config["top_cells_train"]),
            temperature=float(inference_config["temperature"]),
            country_gate=float(inference_config["country_gate"]),
            residual_scale=float(inference_config["residual_scale"]),
            global_blend=float(inference_config["global_blend"]),
        )
        decoded_loss = angular_distance_rad(decoded_xyz, batch["xyz"]).mean()
        contrastive = supervised_contrastive_loss(
            output["embedding"], batch["fine_cell"], float(loss_config["contrastive_temperature"])
        )

        weights = loss_config["weights"]
        components = {
            "fine": fine_loss,
            "coarse": coarse_loss,
            "country": country_loss,
            "global": global_loss,
            "residual": residual_loss,
            "decoded": decoded_loss,
            "contrastive": contrastive,
        }
        total = sum(float(weights[name]) * value for name, value in components.items())
        detached = {name: value.detach() for name, value in components.items()}
        detached["loss"] = total.detach()
        return total, detached
