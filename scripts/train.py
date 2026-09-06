#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from geoguesser.cells import CountryKMeansPartition
from geoguesser.checkpoint import save_checkpoint
from geoguesser.config import apply_overrides, load_config
from geoguesser.data import (
    add_country_ids,
    audit_dataset,
    make_folds,
    make_loader,
    read_labels,
    suggested_num_workers,
)
from geoguesser.engine import (
    evaluate,
    make_optimizer,
    make_warmup_cosine_scheduler,
    train_one_epoch,
)
from geoguesser.losses import GeoCriterion
from geoguesser.leaderboard import leaderboard_fold_assignments
from geoguesser.model import (
    build_model,
    model_config_from_runtime,
    model_parameter_count,
    set_backbone_trainable,
)
from geoguesser.utils import choose_device, hardware_summary, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact geolocation model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--fold", default="0", help="0..K-1, or 'full'")
    parser.add_argument(
        "--folds-csv",
        default=None,
        help="Shared leaderboard folds.csv; overrides split.folds_csv from the config",
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--acknowledge-noncompliant-experiment",
        action="store_true",
        help="Required to use externally pretrained weights; never use those weights for submission without approval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    project_root = Path(config["_config_path"]).parent.parent
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    pretrained = bool(config["model"].get("pretrained", False))
    if pretrained and not args.acknowledge_noncompliant_experiment:
        raise SystemExit(
            "This config uses external pretrained weights and conflicts with the challenge's "
            "training-data rule. Re-run only as a research baseline with "
            "--acknowledge-noncompliant-experiment, or use a scratch/SSL config."
        )

    seed = int(config["seed"])
    seed_everything(seed, bool(config["train"].get("deterministic", False)))
    device = choose_device(args.device)
    amp = device.type == "cuda" and bool(config["train"].get("amp", True))
    workers = suggested_num_workers(config["data"].get("num_workers", "auto"))
    configured_data_root = Path(config["data"]["root"])
    data_root = (
        configured_data_root.resolve()
        if args.data_root is not None or configured_data_root.is_absolute()
        else (project_root / configured_data_root).resolve()
    )
    expected_count = config["data"].get("expected_holdout_count", 2400)
    audit = audit_dataset(
        data_root,
        expected_holdout_count=None if expected_count is None else int(expected_count),
        require_all_countries=bool(config["data"].get("require_all_countries", True)),
    )

    labels, country_to_id = add_country_ids(read_labels(data_root / "train_labels.csv"))
    fold_name = str(args.fold).lower()
    folds_path: Path | None = None
    split_kind = str(config["split"].get("kind", "random"))
    if fold_name == "full":
        train_frame = labels.copy().reset_index(drop=True)
        validation_frame = None
    else:
        fold = int(fold_name)
        folds_csv = args.folds_csv or config["split"].get("folds_csv")
        if folds_csv:
            configured_folds_path = Path(folds_csv)
            folds_path = (
                configured_folds_path.resolve()
                if args.folds_csv is not None or configured_folds_path.is_absolute()
                else (project_root / configured_folds_path).resolve()
            )
            folds = leaderboard_fold_assignments(
                labels, folds_path, n_folds=int(config["split"]["n_folds"])
            )
            split_kind = "predefined"
        elif split_kind == "predefined":
            raise ValueError("split.kind=predefined requires split.folds_csv or --folds-csv")
        else:
            folds = make_folds(
                labels,
                n_splits=int(config["split"]["n_folds"]),
                seed=seed,
                kind=split_kind,
                spatial_target_group_size=int(
                    config["split"].get("spatial_target_group_size", 40)
                ),
            )
        if fold < 0 or fold >= int(config["split"]["n_folds"]):
            raise ValueError(f"Fold {fold} is out of range")
        labels = labels.copy()
        labels["fold"] = folds
        train_frame = labels.loc[labels["fold"] != fold].reset_index(drop=True)
        validation_frame = labels.loc[labels["fold"] == fold].reset_index(drop=True)

    fine_partition = CountryKMeansPartition(
        int(config["geocells"]["fine_target_size"]), seed
    )
    coarse_partition = CountryKMeansPartition(
        int(config["geocells"]["coarse_target_size"]), seed
    )
    train_frame["fine_cell"] = fine_partition.fit_transform(train_frame)
    train_frame["coarse_cell"] = coarse_partition.fit_transform(train_frame)
    if validation_frame is not None:
        validation_frame["fine_cell"] = fine_partition.transform(validation_frame)
        validation_frame["coarse_cell"] = coarse_partition.transform(validation_frame)

    model_config = model_config_from_runtime(
        config,
        n_countries=len(country_to_id),
        n_fine_cells=fine_partition.n_cells_,
        n_coarse_cells=coarse_partition.n_cells_,
    )
    model = build_model(model_config, load_external_weights=True)
    ssl_checkpoint = config["model"].get("backbone_checkpoint")
    if ssl_checkpoint:
        ssl_path = Path(ssl_checkpoint)
        if not ssl_path.is_absolute():
            ssl_path = project_root / ssl_path
        payload = torch.load(ssl_path, map_location="cpu", weights_only=False)
        if payload.get("backbone") != config["model"]["backbone"]:
            raise ValueError("SSL checkpoint backbone does not match model.backbone")
        model.backbone.load_state_dict(payload["backbone_state"], strict=True)
    parameter_count = model_parameter_count(model)
    cap = int(config["rules"]["max_parameters"])
    if parameter_count > cap:
        raise ValueError(f"Model has {parameter_count:,} parameters, above the {cap:,} cap")
    model.to(device)

    configured_output_dir = Path(config["train"]["output_dir"])
    if not configured_output_dir.is_absolute():
        configured_output_dir = project_root / configured_output_dir
    default_run_dir = configured_output_dir / f"fold_{fold_name}"
    run_dir = Path(args.run_dir or default_run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(run_dir / "train_rows.csv", index=False)
    if validation_frame is not None:
        validation_frame.to_csv(run_dir / "validation_rows.csv", index=False)

    criterion = GeoCriterion(fine_partition, coarse_partition, config).to(device)
    ema_decay = float(config["train"].get("ema_decay", 0.0))
    history: list[dict[str, object]] = []
    best_key = (float("inf"), float("inf"))
    best_metrics: dict[str, float] | None = None
    best_checkpoint = run_dir / "best_model.pt"
    final_checkpoint = run_dir / "final_model.pt"
    global_epoch = 0
    start_time = time.perf_counter()
    patience = int(config["train"].get("patience", 9999))
    epochs_without_improvement = 0
    last_image_size = int(config["train"]["stages"][-1]["image_size"])
    last_validation_metrics: dict[str, float] | None = None
    stop_training = False

    for stage_index, stage in enumerate(config["train"]["stages"]):
        image_size = int(stage["image_size"])
        last_image_size = image_size
        batch_size = int(stage["batch_size"])
        freeze_backbone = bool(stage.get("freeze_backbone", False))
        set_backbone_trainable(model, not freeze_backbone)
        train_loader = make_loader(
            train_frame,
            data_root / "train",
            image_size=image_size,
            batch_size=batch_size,
            train=True,
            num_workers=workers,
            rotation_probability=float(config["data"].get("rotation_probability", 0.15)),
            normalization=str(config["model"].get("normalization", "half")),
            seed=seed + stage_index * 1000,
            samples_per_cell=int(config["train"].get("samples_per_cell", 2)),
        )
        validation_loader = None
        if validation_frame is not None:
            validation_loader = make_loader(
                validation_frame,
                data_root / "train",
                image_size=image_size,
                batch_size=int(stage.get("eval_batch_size", batch_size * 2)),
                train=False,
                num_workers=workers,
                rotation_probability=0.0,
                normalization=str(config["model"].get("normalization", "half")),
                seed=seed,
            )
        optimizer = make_optimizer(
            model,
            learning_rate=float(stage["learning_rate"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        accumulation = int(stage.get("accumulation_steps", 1))
        updates_per_epoch = max(1, (len(train_loader) + accumulation - 1) // accumulation)
        total_steps = int(stage["epochs"]) * updates_per_epoch
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(float(config["train"]["warmup_fraction"]) * total_steps),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        for stage_epoch in range(int(stage["epochs"])):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                global_epoch,
                accumulation,
                float(config["train"]["grad_clip"]),
                amp,
            )
            record: dict[str, object] = {
                "epoch": global_epoch,
                "stage": stage_index,
                "stage_epoch": stage_epoch,
                "image_size": image_size,
                "freeze_backbone": freeze_backbone,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }
            if validation_loader is not None:
                validation_metrics, validation_output = evaluate(
                    model,
                    validation_loader,
                    fine_partition,
                    config["inference"],
                    device,
                )
                last_validation_metrics = validation_metrics
                record.update({f"val_{key}": value for key, value in validation_metrics.items()})
                key = (validation_metrics["median_km"], validation_metrics["mean_km"])
                if key < best_key:
                    best_key = key
                    best_metrics = validation_metrics
                    epochs_without_improvement = 0
                    save_checkpoint(
                        best_checkpoint,
                        model=model,
                        model_config=model_config,
                        fine_partition=fine_partition,
                        coarse_partition=coarse_partition,
                        country_to_id=country_to_id,
                        config=config,
                        metadata={
                            "fold": fold_name,
                            "epoch": global_epoch,
                            "image_size": image_size,
                            "metrics": validation_metrics,
                        },
                    )
                    pd.DataFrame(
                        {
                            "filename": validation_output["filename"],
                            "true_lat": validation_output["truth"][:, 0],
                            "true_lng": validation_output["truth"][:, 1],
                            "pred_lat": validation_output["prediction"][:, 0],
                            "pred_lng": validation_output["prediction"][:, 1],
                        }
                    ).to_csv(run_dir / "oof_predictions.csv", index=False)
                else:
                    epochs_without_improvement += 1
            record["epoch_seconds"] = time.perf_counter() - epoch_start
            history.append(record)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            print(json.dumps(record, sort_keys=True))
            global_epoch += 1
            if validation_loader is not None and epochs_without_improvement >= patience:
                print(f"Early stopping after {epochs_without_improvement} non-improving epochs")
                stop_training = True
                break
        if stop_training:
            break

    elapsed = time.perf_counter() - start_time
    save_checkpoint(
        final_checkpoint,
        model=model,
        model_config=model_config,
        fine_partition=fine_partition,
        coarse_partition=coarse_partition,
        country_to_id=country_to_id,
        config=config,
        metadata={
            "fold": fold_name,
            "epoch": global_epoch - 1,
            "image_size": last_image_size,
            "metrics": last_validation_metrics or {},
        },
    )
    summary = {
        "config": str(Path(args.config).resolve()),
        "data_root": str(data_root),
        "fold": fold_name,
        "split_kind": split_kind if fold_name != "full" else "full",
        "folds_csv": None if folds_path is None else str(folds_path),
        "train_rows": len(train_frame),
        "validation_rows": 0 if validation_frame is None else len(validation_frame),
        "fine_cells": fine_partition.n_cells_,
        "coarse_cells": coarse_partition.n_cells_,
        "parameter_count": parameter_count,
        "parameter_cap": cap,
        "rule_compliant": not pretrained,
        "ema_decay": ema_decay,
        "best_metrics": best_metrics,
        "elapsed_seconds": elapsed,
        "checkpoint": str(best_checkpoint if best_checkpoint.exists() else final_checkpoint),
        "audit": audit,
        "hardware": hardware_summary(device),
    }
    write_json(summary, run_dir / "run_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
