#!/usr/bin/env python3
"""Verify a materialized UFPR04 subset against its tracked manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from scripts.inventory_pklot import WEATHERS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.metadata.json"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_inventory.csv"),
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("data/processed/ufpr04_balanced_v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    errors: list[str] = []

    if sha256_file(args.manifest) != metadata["source"]["manifest_sha256"]:
        errors.append("manifest SHA-256 does not match metadata")
    if sha256_file(args.inventory) != metadata["source"]["inventory_sha256"]:
        errors.append("inventory SHA-256 does not match metadata")

    expected_per_weather = (
        metadata["selection"]["dates_per_weather"] * metadata["selection"]["images_per_date"]
    )
    weather_counts = Counter(row["weather"] for row in rows)
    for weather in WEATHERS:
        if weather_counts[weather] != expected_per_weather:
            errors.append(
                f"{weather}: expected {expected_per_weather} images, "
                f"found {weather_counts[weather]}"
            )

    expected_files: set[Path] = set()
    by_weather_date: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for row in rows:
        for path_field, hash_field in (
            ("image_relpath", "image_sha256"),
            ("annotation_relpath", "annotation_sha256"),
        ):
            relative = Path(row[path_field])
            expected_files.add(relative)
            path = args.subset_root / relative
            if not path.is_file():
                errors.append(f"missing materialized file: {relative.as_posix()}")
            elif sha256_file(path) != row[hash_field]:
                errors.append(f"hash mismatch: {relative.as_posix()}")

        timestamp = datetime.fromisoformat(f"{row['acquisition_date']}T{row['capture_time']}")
        by_weather_date[(row["weather"], row["acquisition_date"])].append(timestamp)

    actual_files = {
        path.relative_to(args.subset_root) for path in args.subset_root.rglob("*") if path.is_file()
    }
    for unexpected in sorted(actual_files - expected_files):
        errors.append(f"unexpected materialized file: {unexpected.as_posix()}")

    minimum_gap_seconds = metadata["selection"]["minimum_gap_minutes"] * 60
    for key, timestamps in sorted(by_weather_date.items()):
        ordered = sorted(timestamps)
        for first, second in zip(ordered, ordered[1:]):
            if (second - first).total_seconds() < minimum_gap_seconds:
                errors.append(f"minimum gap violation: {key[0]}/{key[1]}")
                break

    summary = {
        "errors": errors,
        "materialized_files": len(actual_files),
        "selected_images": len(rows),
        "weather_counts": {weather: weather_counts[weather] for weather in WEATHERS},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
