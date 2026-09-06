#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from tqdm.auto import tqdm

from geoguesser.data import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate files and inspect image metadata")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--expected-holdout-count", type=int, default=2400)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--allow-country-subset", action="store_true")
    args = parser.parse_args()
    result = audit_dataset(
        args.data_root,
        args.expected_holdout_count,
        require_all_countries=not args.allow_country_subset,
    )
    paths = sorted((args.data_root / "train").glob("*.jpg"))[: args.sample]
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    exif_oriented = 0
    for path in tqdm(paths, desc="inspect images"):
        with Image.open(path) as image:
            sizes[image.size] += 1
            modes[image.mode] += 1
            transposed = ImageOps.exif_transpose(image)
            exif_oriented += int(transposed.size != image.size)
    result["inspected_images"] = len(paths)
    result["sizes"] = {f"{w}x{h}": count for (w, h), count in sizes.most_common()}
    result["modes"] = dict(modes)
    result["exif_axis_swaps"] = exif_oriented
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

