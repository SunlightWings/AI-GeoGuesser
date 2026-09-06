#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geoguesser.advanced_decoders import (
    build_intersection_partition,
    decode_cplanet_numpy,
    decode_fine_numpy,
)
from geoguesser.frozen_geoclip import metric_summary


OUTPUT_KEYS = ("fine_logits", "coarse_logits", "country_logits", "global_xyz", "fine_residuals")


def comma_values(text: str, cast):
    return [cast(item) for item in text.split(",") if item.strip()]


def read_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(OUTPUT_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Cache {path} is missing arrays: {sorted(missing)}")
        arrays = {key: archive[key].astype(np.float64) for key in OUTPUT_KEYS}
        arrays["truth_latlng"] = archive["truth_latlng"].astype(np.float64)
        arrays["fine_centers_xyz"] = archive["fine_centers_xyz"].astype(np.float64)
        arrays["fine_cell_country_ids"] = archive["fine_cell_country_ids"].astype(np.int64)
        metadata = json.loads(str(archive["metadata"].item()))
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-training fine-decoder and CPlaNet intersection ablations"
    )
    parser.add_argument("--caches", nargs="+", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selected-config", default=None, type=Path)
    parser.add_argument("--top-cells", default="1,3,5,8")
    parser.add_argument("--temperatures", default="0.65,0.85,1.0,1.2")
    parser.add_argument("--country-gates", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--residual-scales", default="0.0,0.01,0.025,0.05")
    parser.add_argument("--global-blends", default="0.0,0.05,0.10")
    parser.add_argument("--coarse-weights", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--intersection-top-k", default="1,3,5")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    required = {"lat", "lng", "country_id", "fine_cell", "coarse_cell"}
    if missing := required - set(train.columns):
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    intersection = build_intersection_partition(
        train[["lat", "lng"]].to_numpy(),
        train["fine_cell"].to_numpy(),
        train["coarse_cell"].to_numpy(),
        train["country_id"].to_numpy(),
    )
    caches = [(*read_cache(path), path) for path in args.caches]
    hashes = {metadata["checkpoint_sha256"] for _, metadata, _ in caches}
    if len(hashes) != 1:
        raise ValueError("Every cache must come from the same checkpoint")
    truth = caches[0][0]["truth_latlng"]
    if any(not np.array_equal(item[0]["truth_latlng"], truth) for item in caches[1:]):
        raise ValueError("Cache truth rows differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.selected_config is not None:
        selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
        matching = [item for item in caches if item[1]["tta"] == selected["tta"]]
        if len(matching) != 1:
            raise ValueError(f"No unique cache exists for selected TTA {selected['tta']}")
        output, metadata, path = matching[0]
        prediction = decode_cplanet_numpy(
            output,
            intersection,
            top_intersections=int(selected["top_intersections"]),
            temperature=float(selected["temperature"]),
            country_gate=float(selected["country_gate"]),
            coarse_weight=float(selected["coarse_weight"]),
        )
        row = {
            "family": "cplanet_confirmation",
            "cache": str(path),
            "tta": metadata["tta"],
            "intersections": len(intersection.center_xyz),
            **{
                key: selected[key]
                for key in ("top_intersections", "temperature", "country_gate", "coarse_weight")
            },
            **metric_summary(truth, prediction),
        }
        pd.DataFrame([row]).to_csv(args.output_dir / "cplanet_confirmation.csv", index=False)
        print(json.dumps(row, indent=2))
        return

    top_cells = comma_values(args.top_cells, int)
    temperatures = comma_values(args.temperatures, float)
    gates = comma_values(args.country_gates, float)
    residuals = comma_values(args.residual_scales, float)
    blends = comma_values(args.global_blends, float)
    coarse_weights = comma_values(args.coarse_weights, float)
    intersection_top_k = comma_values(args.intersection_top_k, int)
    baseline_rows: list[dict[str, object]] = []
    cplanet_rows: list[dict[str, object]] = []

    for output, metadata, path in caches:
        common = {"cache": str(path), "tta": metadata["tta"]}
        for top_k, temperature, gate, residual, blend in itertools.product(
            top_cells, temperatures, gates, residuals, blends
        ):
            prediction = decode_fine_numpy(
                output,
                output["fine_centers_xyz"],
                output["fine_cell_country_ids"],
                top_cells=top_k,
                temperature=temperature,
                country_gate=gate,
                residual_scale=residual,
                global_blend=blend,
            )
            baseline_rows.append(
                {
                    "family": "fine_decoder",
                    **common,
                    "top_cells": top_k,
                    "temperature": temperature,
                    "country_gate": gate,
                    "residual_scale": residual,
                    "global_blend": blend,
                    **metric_summary(truth, prediction),
                }
            )
        for top_k, temperature, gate, coarse_weight in itertools.product(
            intersection_top_k, temperatures, gates, coarse_weights
        ):
            prediction = decode_cplanet_numpy(
                output,
                intersection,
                top_intersections=top_k,
                temperature=temperature,
                country_gate=gate,
                coarse_weight=coarse_weight,
            )
            cplanet_rows.append(
                {
                    "family": "cplanet",
                    **common,
                    "intersections": len(intersection.center_xyz),
                    "top_intersections": top_k,
                    "temperature": temperature,
                    "country_gate": gate,
                    "coarse_weight": coarse_weight,
                    **metric_summary(truth, prediction),
                }
            )

    baseline = pd.DataFrame(baseline_rows).sort_values(["median_km", "mean_km"])
    cplanet = pd.DataFrame(cplanet_rows).sort_values(["median_km", "mean_km"])
    baseline.to_csv(args.output_dir / "fine_decoder_ablation.csv", index=False)
    cplanet.to_csv(args.output_dir / "cplanet_ablation.csv", index=False)
    selected = cplanet.iloc[0].to_dict()
    selected_path = args.output_dir / "cplanet_selected.json"
    selected_path.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Best fine decoder:")
    print(baseline.head(10).to_string(index=False))
    print("\nBest CPlaNet decoder:")
    print(cplanet.head(10).to_string(index=False))
    print(f"\nSelected CPlaNet configuration: {selected_path.resolve()}")


if __name__ == "__main__":
    main()
