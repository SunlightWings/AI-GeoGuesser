#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from geoguesser.config import apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential cross-validation folds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--folds-csv", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--acknowledge-noncompliant-experiment", action="store_true")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    project_root = Path(config["_config_path"]).parent.parent
    folds = args.folds or list(range(int(config["split"]["n_folds"])))
    if len(set(folds)) != len(folds):
        raise ValueError("Fold IDs must not be repeated")
    invalid = sorted(set(folds) - set(range(int(config["split"]["n_folds"]))))
    if invalid:
        raise ValueError(f"Fold IDs out of range: {invalid}")
    folds_csv = args.folds_csv or config["split"].get("folds_csv")
    if str(config["split"].get("kind", "random")) == "predefined" and not folds_csv:
        raise ValueError("split.kind=predefined requires split.folds_csv or --folds-csv")
    configured_output_root = Path(config["train"]["output_dir"])
    output_root = (
        configured_output_root.resolve()
        if configured_output_root.is_absolute()
        else (project_root / configured_output_root).resolve()
    )
    if folds_csv:
        configured_folds_path = Path(folds_csv)
        folds_csv = str(
            configured_folds_path.resolve()
            if args.folds_csv is not None or configured_folds_path.is_absolute()
            else (project_root / configured_folds_path).resolve()
        )
    summaries = []
    for fold in folds:
        command = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--config",
            args.config,
            "--fold",
            str(fold),
            "--device",
            args.device,
        ]
        if args.data_root:
            command.extend(["--data-root", args.data_root])
        if folds_csv:
            command.extend(["--folds-csv", str(folds_csv)])
        for override in args.overrides:
            command.extend(["--set", override])
        if args.acknowledge_noncompliant_experiment:
            command.append("--acknowledge-noncompliant-experiment")
        subprocess.run(command, check=True)
        summary_path = output_root / f"fold_{fold}" / "run_summary.json"
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    rows = []
    for summary in summaries:
        row = {"fold": summary["fold"], "parameter_count": summary["parameter_count"]}
        row.update(summary["best_metrics"] or {})
        rows.append(row)
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "cross_validation.csv", index=False)
    numeric = [column for column in frame.columns if column != "fold"]
    aggregate = {
        column: {"mean": float(frame[column].mean()), "std": float(frame[column].std(ddof=0))}
        for column in numeric
    }
    (output_root / "cross_validation_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(aggregate, indent=2))

    expected_folds = set(range(int(config["split"]["n_folds"])))
    if set(folds) == expected_folds and folds_csv:
        evaluation_command = [
            sys.executable,
            str(Path(__file__).with_name("evaluate_leaderboard.py")),
            "--run-dir",
            str(output_root),
            "--leaderboard-csv",
            str(Path(folds_csv).resolve()),
            "--output",
            str(output_root / "oof_predictions.csv"),
            "--metrics-output",
            str(output_root / "leaderboard_metrics.json"),
        ]
        subprocess.run(evaluation_command, check=True)
    elif set(folds) != expected_folds:
        print("Pooled leaderboard metrics skipped because only a subset of folds was run.")


if __name__ == "__main__":
    main()
