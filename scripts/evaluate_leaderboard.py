#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from geoguesser.leaderboard import (
    collect_oof_predictions,
    read_leaderboard_folds,
    score_oof_predictions,
    validate_prediction_frame,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score pooled out-of-fold predictions using the shared unofficial-leaderboard folds. "
            "This script never runs one checkpoint over rows that checkpoint trained on."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Cross-validation directory containing fold_0/.../fold_4/oof_predictions.csv",
    )
    source.add_argument(
        "--predictions",
        type=Path,
        help="An already pooled CSV with filename,pred_lat,pred_lng",
    )
    parser.add_argument("--leaderboard-csv", required=True, type=Path)
    parser.add_argument("--output", default="leaderboard/oof_predictions.csv", type=Path)
    parser.add_argument("--metrics-output", default=None, type=Path)
    return parser.parse_args()


def checkpoint_metadata(run_dir: Path, folds: pd.DataFrame) -> dict[str, object]:
    summaries = []
    for fold in sorted(folds["fold"].unique().tolist()):
        path = run_dir / f"fold_{fold}" / "run_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if str(summary.get("fold")) != str(fold):
            raise ValueError(f"Fold summary has the wrong fold ID: {path}")
        if not bool(summary.get("rule_compliant", False)):
            raise ValueError(f"Fold {fold} is marked non-compliant")
        if summary.get("split_kind") != "predefined" or not summary.get("folds_csv"):
            raise ValueError(
                f"Fold {fold} was not trained with a recorded predefined folds.csv"
            )
        expected_rows = int((folds["fold"] == fold).sum())
        if int(summary.get("validation_rows", -1)) != expected_rows:
            raise ValueError(
                f"Fold {fold} reports {summary.get('validation_rows')} validation rows; "
                f"expected {expected_rows}"
            )
        summaries.append(summary)
    parameter_counts = {int(item["parameter_count"]) for item in summaries}
    if len(parameter_counts) != 1:
        raise ValueError(f"Fold models have inconsistent parameter counts: {sorted(parameter_counts)}")
    parameter_count = parameter_counts.pop()
    if parameter_count > 5_000_000:
        raise ValueError(f"Fold model has {parameter_count:,} parameters, above the 5M cap")
    return {"parameter_count": parameter_count, "rule_compliant": True}


def print_report(report: dict[str, object]) -> None:
    overall = report["overall"]
    assert isinstance(overall, dict)
    print(f"Overall median_km: {overall['median_km']:.4f}")
    print(f"Overall mean_km: {overall['mean_km']:.4f}")
    print(f"Overall <200km %: {100.0 * overall['lt_200_km']:.2f}")
    print(f"Overall <750km %: {100.0 * overall['lt_750_km']:.2f}")
    print("\nPer-fold:")
    per_fold = report["per_fold"]
    assert isinstance(per_fold, dict)
    for fold, metrics in per_fold.items():
        print(
            f"fold {fold}: median={metrics['median_km']:.4f} "
            f"mean={metrics['mean_km']:.4f} "
            f"<200%={100.0 * metrics['lt_200_km']:.2f} "
            f"<750%={100.0 * metrics['lt_750_km']:.2f}"
        )


def main() -> None:
    args = parse_args()
    folds = read_leaderboard_folds(args.leaderboard_csv)
    if args.run_dir is not None:
        predictions = collect_oof_predictions(args.run_dir, folds)
        metadata = checkpoint_metadata(args.run_dir, folds)
    else:
        predictions = validate_prediction_frame(
            pd.read_csv(args.predictions), set(folds["filename"])
        )
        metadata = {}

    report, _ = score_oof_predictions(folds, predictions)
    report.update(metadata)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, encoding="utf-8", float_format="%.8f")
    metrics_output = (args.metrics_output or output.with_suffix(".metrics.json")).resolve()
    write_metrics(report, metrics_output)
    print_report(report)
    if metadata:
        print(f"\nParameters per fold model: {metadata['parameter_count']:,}")
    print(f"Pooled OOF predictions: {output}")
    print(f"Metrics JSON: {metrics_output}")


if __name__ == "__main__":
    main()
