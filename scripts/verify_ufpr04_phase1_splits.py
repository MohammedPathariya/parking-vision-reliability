#!/usr/bin/env python3
"""Verify the tracked date-disjoint Phase 1 UFPR04 split manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.inventory_pklot import sha256_file
from scripts.select_ufpr04_subset import SUBSET_FIELDS
from scripts.split_ufpr04_phase1 import SPLIT_NAMES, validate_splits


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBSET_FIELDS:
            raise ValueError(f"Unexpected manifest schema in {path}")
        return list(reader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_splits_v1.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_splits_v1.metadata.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_manifest(args.source_manifest)
    split_rows = read_manifest(args.manifest)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    errors: list[str] = []

    try:
        validate_splits(split_rows, source_rows)
    except ValueError as exc:
        errors.append(str(exc))

    if sha256_file(args.source_manifest) != metadata["source_manifest_sha256"]:
        errors.append("source manifest SHA-256 does not match metadata")
    if sha256_file(args.manifest) != metadata["split_manifest_sha256"]:
        errors.append("split manifest SHA-256 does not match metadata")

    for split in SPLIT_NAMES:
        path = args.manifest.with_name(f"{args.manifest.stem}_{split}.csv")
        if not path.is_file():
            errors.append(f"missing individual split manifest: {path.name}")
            continue
        expected = [row for row in split_rows if row["split"] == split]
        if read_manifest(path) != expected:
            errors.append(f"individual split manifest does not match: {path.name}")

    print(
        json.dumps(
            {
                "errors": errors,
                "source_images": len(source_rows),
                "split_images": len(split_rows),
                "splits": {
                    split: sum(row["split"] == split for row in split_rows)
                    for split in SPLIT_NAMES
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
