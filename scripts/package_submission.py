#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final course submission zip")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path, help="Completed .tex or PDF")
    parser.add_argument("--reference-bank", type=Path, default=None)
    parser.add_argument("--acknowledge-reference-bank-rule", action="store_true")
    parser.add_argument("--output", default="submission.zip", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(bundle["parameter_count"]) > 5_000_000:
        raise ValueError("Checkpoint exceeds the 5M parameter cap")
    if not bundle.get("rule_compliant", False):
        raise ValueError("Checkpoint is tagged noncompliant (external pretraining)")
    predictions = pd.read_csv(args.predictions)
    if predictions.columns.tolist() != ["filename", "pred_lat", "pred_lng"] or len(predictions) != 2400:
        raise ValueError("Predictions do not satisfy the basic 2,400-row CSV contract")
    if "TODO" in args.report.read_text(encoding="utf-8", errors="ignore"):
        raise ValueError("The write-up still contains TODO placeholders")
    if args.reference_bank is not None and not args.acknowledge_reference_bank_rule:
        raise ValueError("Reference bank supplied without explicit rule acknowledgement")

    with tempfile.TemporaryDirectory(prefix="geolocation_submission_") as temporary:
        staging = Path(temporary) / "submission"
        staging.mkdir()
        for relative in ("src", "scripts", "configs"):
            shutil.copytree(root / relative, staging / relative, ignore=shutil.ignore_patterns("__pycache__"))
        for relative in ("README.md", "EXPERIMENTS_README.md", "requirements.txt", "pyproject.toml"):
            shutil.copy2(root / relative, staging / relative)
        shutil.copy2(args.checkpoint, staging / "model.pt")
        shutil.copy2(args.predictions, staging / "predictions.csv")
        shutil.copy2(args.report, staging / args.report.name)
        if args.reference_bank is not None:
            shutil.copy2(args.reference_bank, staging / "reference_bank.npz")
        manifest = {
            "parameter_count": int(bundle["parameter_count"]),
            "rule_compliant": bool(bundle["rule_compliant"]),
            "model_sha256": sha256(staging / "model.pt"),
            "predictions_sha256": sha256(staging / "predictions.csv"),
        }
        if args.reference_bank is not None:
            manifest["reference_bank_sha256"] = sha256(staging / "reference_bank.npz")
        (staging / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging.parent))
    print(json.dumps({"output": str(args.output.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
