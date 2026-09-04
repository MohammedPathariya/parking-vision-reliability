#!/usr/bin/env python3
"""Verify a selectively downloaded UFPR04 subset against its frozen manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.download_ufpr04_subset import ANNOTATION_ARCHIVE_SHA256, validate_annotation_coverage
from scripts.inventory_pklot import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.csv"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/PKLot/UFPR04_selected_v1"))
    return parser.parse_args()


def verify_download(manifest: Path, raw_root: Path) -> dict[str, object]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    downloaded = 0
    for row in rows:
        path = raw_root / row["image_relpath"]
        if not path.is_file():
            errors.append(f"missing image: {row['image_relpath']}")
        elif sha256_file(path) != row["image_sha256"]:
            errors.append(f"hash mismatch: {row['image_relpath']}")
        else:
            downloaded += 1

    annotation_dir = raw_root / "_annotations"
    archive_path = annotation_dir / "ufpr04_spots.tar.xz"
    annotation_path = annotation_dir / "ufpr04_spots.json"
    if not archive_path.is_file():
        errors.append("missing official annotation archive")
    elif sha256_file(archive_path) != ANNOTATION_ARCHIVE_SHA256:
        errors.append("annotation archive hash mismatch")
    if not annotation_path.is_file():
        errors.append("missing extracted annotation JSON")
    else:
        try:
            validate_annotation_coverage(annotation_path, rows)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"annotation coverage check failed: {error}")

    return {"errors": errors, "verified_images": downloaded, "selected_images": len(rows)}


def main() -> int:
    args = parse_args()
    summary = verify_download(args.manifest, args.raw_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
