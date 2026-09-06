#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(
        json.dumps(
            {
                "format_version": bundle.get("format_version"),
                "parameter_count": bundle.get("parameter_count"),
                "rule_compliant": bundle.get("rule_compliant"),
                "external_pretraining": bundle.get("external_pretraining"),
                "model_config": bundle.get("model_config"),
                "metadata": bundle.get("metadata"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
