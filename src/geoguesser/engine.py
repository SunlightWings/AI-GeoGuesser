from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .cells import CountryKMeansPartition
from .geo import metric_summary
from .inference import forward_tta
from .losses import GeoCriterion
from .model import CompactGeoModel, decode_outputs
from .utils import move_to_device


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "log_scale" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def make_warmup_cosine_scheduler(
    optimizer: Optimizer, total_steps: int, warmup_steps: int
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps - 1)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def train_one_epoch(
    model: CompactGeoModel,
    loader: DataLoader,
    criterion: GeoCriterion,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    accumulation_steps: int,
    grad_clip: float,
    amp: bool,
) -> dict[str, float]:
    model.train()
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)
    totals: defaultdict[str, float] = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train epoch {epoch:03d}", leave=False)
    for step, batch in enumerate(progress):
        batch = move_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(batch["image"])
            loss, components = criterion(output, batch)
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        update = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
        if update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        for name, value in components.items():
            totals[name] += float(value)
        progress.set_postfix(loss=f"{float(loss):.3f}")
    return {name: total / max(1, len(loader)) for name, total in totals.items()}


@torch.inference_mode()
def evaluate(
    model: CompactGeoModel,
    loader: DataLoader,
    partition: CountryKMeansPartition,
    inference_config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    filenames: list[str] = []
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    predicted_countries: list[np.ndarray] = []
    true_countries: list[np.ndarray] = []
    for batch in tqdm(loader, desc="validate", leave=False):
        batch = move_to_device(batch, device)
        output = forward_tta(
            model,
            batch["image"],
            str(inference_config.get("tta", "none")),
        )
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
        truths.append(batch["latlng"].cpu().numpy())
        predicted_countries.append(output["country_logits"].argmax(dim=1).cpu().numpy())
        true_countries.append(batch["country_id"].cpu().numpy())
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    metrics = metric_summary(truth, prediction)
    metrics["country_accuracy"] = float(
        np.mean(np.concatenate(predicted_countries) == np.concatenate(true_countries))
    )
    return metrics, {
        "filename": np.asarray(filenames),
        "prediction": prediction,
        "truth": truth,
    }
