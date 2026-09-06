#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark several configs on the same fold")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--acknowledge-noncompliant-experiment", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_path in args.configs:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        name = str(config.get("experiment_name", Path(config_path).stem))
        run_dir = output_dir / name / f"fold_{args.fold}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--config",
            config_path,
            "--data-root",
            args.data_root,
            "--fold",
            str(args.fold),
            "--run-dir",
            str(run_dir),
            "--device",
            args.device,
        ]
        for override in args.overrides:
            command.extend(["--set", override])
        if args.acknowledge_noncompliant_experiment:
            command.append("--acknowledge-noncompliant-experiment")
        start = time.perf_counter()
        try:
            subprocess.run(command, check=True)
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            row = {
                "experiment": name,
                "status": "ok",
                "parameter_count": summary["parameter_count"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "rule_compliant": summary["rule_compliant"],
                **(summary["best_metrics"] or {}),
            }
        except subprocess.CalledProcessError as error:
            row = {
                "experiment": name,
                "status": f"failed:{error.returncode}",
                "elapsed_seconds": time.perf_counter() - start,
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "benchmark_results.csv", index=False)
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

