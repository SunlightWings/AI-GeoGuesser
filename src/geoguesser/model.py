from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)

from .cells import CountryKMeansPartition
from .geo import unit_to_latlng_torch


class SignedGeM(nn.Module):
    """Generalized-mean pooling that remains valid for signed activations."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(1.0, 6.0)
        positive = F.adaptive_avg_pool2d(features.clamp_min(self.eps).pow(p), 1).pow(1.0 / p)
        negative = F.adaptive_avg_pool2d((-features).clamp_min(self.eps).pow(p), 1).pow(1.0 / p)
        return (positive - negative).flatten(1)


class CosineHead(nn.Module):
    def __init__(self, embedding_dim: int, classes: int, initial_scale: float = 16.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp().clamp(1.0, 64.0)
        return scale * F.linear(F.normalize(embedding, dim=1), F.normalize(self.weight, dim=1))


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)
        return base.features, int(base.classifier[1].in_features)
    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        return base.features, int(base.classifier[0].in_features)
    raise ValueError(f"Unsupported backbone: {name}")


class CompactGeoModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        embedding_dim: int,
        dropout: float,
        n_countries: int,
        n_fine_cells: int,
        n_coarse_cells: int,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feature_dim = build_backbone(backbone, pretrained)
        self.pool = SignedGeM()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.country_head = CosineHead(embedding_dim, n_countries)
        self.fine_head = CosineHead(embedding_dim, n_fine_cells)
        self.coarse_head = CosineHead(embedding_dim, n_coarse_cells)
        self.global_xyz_head = nn.Linear(embedding_dim, 3)
        self.fine_residual_head = nn.Linear(embedding_dim, n_fine_cells * 3)
        self.n_fine_cells = int(n_fine_cells)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(image)
        hidden = self.projection(self.pool(feature_map))
        embedding = F.normalize(hidden, dim=1)
        return {
            "embedding": embedding,
            "country_logits": self.country_head(embedding),
            "fine_logits": self.fine_head(embedding),
            "coarse_logits": self.coarse_head(embedding),
            "global_xyz": F.normalize(self.global_xyz_head(hidden), dim=1),
            "fine_residuals": self.fine_residual_head(hidden).view(-1, self.n_fine_cells, 3),
        }


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_config_from_runtime(
    config: dict[str, Any], n_countries: int, n_fine_cells: int, n_coarse_cells: int
) -> dict[str, Any]:
    return {
        "backbone": config["model"]["backbone"],
        "pretrained": bool(config["model"].get("pretrained", False)),
        "embedding_dim": int(config["model"]["embedding_dim"]),
        "dropout": float(config["model"]["dropout"]),
        "n_countries": int(n_countries),
        "n_fine_cells": int(n_fine_cells),
        "n_coarse_cells": int(n_coarse_cells),
    }


def build_model(model_config: dict[str, Any], load_external_weights: bool = True) -> CompactGeoModel:
    runtime = dict(model_config)
    if not load_external_weights:
        runtime["pretrained"] = False
    return CompactGeoModel(**runtime)


def decode_outputs(
    output: dict[str, torch.Tensor],
    partition: CountryKMeansPartition,
    top_cells: int = 1,
    temperature: float = 1.0,
    country_gate: float = 2.0,
    residual_scale: float = 0.025,
    global_blend: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a multimodal cell distribution into one spherical coordinate."""
    device = output["fine_logits"].device
    centers = torch.as_tensor(partition.output_centers_xyz_, dtype=torch.float32, device=device)
    cell_countries = torch.as_tensor(partition.cell_country_ids_, dtype=torch.long, device=device)
    country_log_probability = F.log_softmax(output["country_logits"], dim=1)
    score = output["fine_logits"] / temperature + country_gate * country_log_probability[:, cell_countries]
    k = min(max(1, int(top_cells)), score.shape[1])
    top_score, top_index = score.topk(k, dim=1)
    probability = F.softmax(top_score, dim=1)

    selected_centers = centers[top_index]
    row = torch.arange(len(score), device=device)[:, None]
    selected_residuals = output["fine_residuals"][row, top_index]
    candidate_xyz = F.normalize(
        selected_centers + residual_scale * torch.tanh(selected_residuals), dim=2
    )
    cell_xyz = F.normalize((probability[..., None] * candidate_xyz).sum(dim=1), dim=1)
    prediction_xyz = F.normalize(
        (1.0 - global_blend) * cell_xyz + global_blend * output["global_xyz"], dim=1
    )
    return unit_to_latlng_torch(prediction_xyz), prediction_xyz


def set_backbone_trainable(model: CompactGeoModel, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
