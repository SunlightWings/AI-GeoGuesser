#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise train -> checkpoint -> predict -> validate")
    parser.add_argument("--attachments", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    attachments = args.attachments.resolve()
    labels = args.labels.resolve()
    smoke_data = root / "outputs" / "smoke_data"
    smoke_run = root / "outputs" / "smoke_run"
    predictions = smoke_run / "predictions.csv"
    run(
        [sys.executable, "scripts/make_smoke_dataset.py", "--images", str(attachments), "--labels", str(labels), "--output", str(smoke_data)],
        root,
    )
    run(
        [sys.executable, "scripts/train.py", "--config", "configs/smoke.yaml", "--data-root", str(smoke_data), "--fold", "full", "--run-dir", str(smoke_run), "--device", args.device],
        root,
    )
    holdout_count = len(list((smoke_data / "holdout_public").glob("*.jpg")))
    run(
        [sys.executable, "scripts/predict.py", "--checkpoint", str(smoke_run / "final_model.pt"), "--data-root", str(smoke_data), "--output", str(predictions), "--expected-count", str(holdout_count), "--batch-size", "2", "--num-workers", "0", "--device", args.device],
        root,
    )
    run(
        [sys.executable, "scripts/validate_submission.py", "--predictions", str(predictions), "--holdout-dir", str(smoke_data / "holdout_public"), "--expected-count", str(holdout_count)],
        root,
    )
    print(f"END-TO-END SMOKE TEST PASSED: {predictions}")


if __name__ == "__main__":
    main()
