#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SETTING_COLUMNS = (
    "tta", "top_cells", "temperature", "country_gate", "residual_scale", "global_blend"
)
METRIC_COLUMNS = ("median_km", "mean_km", "lt_200_km", "lt_750_km", "p90_km")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed decoder grid")
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.grid)
    required = set(SETTING_COLUMNS + METRIC_COLUMNS)
    if missing := required - set(frame.columns):
        raise ValueError(f"Decoder grid is missing columns: {sorted(missing)}")
    frame = frame.sort_values(["median_km", "mean_km"]).reset_index(drop=True)
    best = frame.iloc[0]
    one_factor = []
    for factor in SETTING_COLUMNS:
        mask = pd.Series(True, index=frame.index)
        for setting in SETTING_COLUMNS:
            if setting != factor:
                mask &= frame[setting].eq(best[setting])
        subset = frame.loc[mask, list(SETTING_COLUMNS + METRIC_COLUMNS)].copy()
        subset.insert(0, "ablated_factor", factor)
        one_factor.append(subset)
    result = pd.concat(one_factor, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "one_factor_around_best.csv", index=False)
    frame.head(25).to_csv(args.output_dir / "top_25_decoder_settings.csv", index=False)
    payload = {
        key: best[key].item() if hasattr(best[key], "item") else best[key]
        for key in SETTING_COLUMNS + METRIC_COLUMNS
    }
    (args.output_dir / "best_decoder.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
