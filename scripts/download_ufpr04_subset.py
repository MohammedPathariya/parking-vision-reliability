#!/usr/bin/env python3
"""Download the manifest-selected UFPR04 frames and their official annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from scripts.inventory_pklot import sha256_file

IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/DSBD-Research/MetaPKLot-Dataset/main/PKLot/UFPR04"
)
ANNOTATION_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/DSBD-Research/MetaPKLot-Dataset/main/"
    "annotations/original/spots/PKLot/ufpr04_spots.tar.xz"
)
ANNOTATION_ARCHIVE_SHA256 = "e7fc2c3151e86baa25de034e41a27bf6637a3990f314ff8db1079cf2ac850227"
ANNOTATION_MEMBER = "ufpr04_spots.json"


def safe_destination(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Destination escapes raw root: {relative}")
    return destination


def validate_raw_root(raw_root: Path, data_root: Path = Path("data/raw")) -> Path:
    resolved_root = raw_root.resolve()
    if not resolved_root.is_relative_to(data_root.resolve()):
        raise ValueError(f"--raw-root must remain under {data_root}")
    return resolved_root


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_relpath", "image_sha256", "annotation_relpath"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest is missing required fields: {path}")
    image_paths = [row["image_relpath"] for row in rows]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Manifest contains duplicate image paths")
    for row in rows:
        relative = Path(row["image_relpath"])
        if relative.parts[:1] not in {("Sunny",), ("Cloudy",), ("Rainy",)}:
            raise ValueError(f"Unexpected UFPR04 image path: {relative}")
        if relative.suffix.lower() != ".jpg" or len(row["image_sha256"]) != 64:
            raise ValueError(f"Invalid image checksum entry: {relative}")
    return rows


def download_file(url: str, destination: Path, expected_sha256: str | None) -> str:
    """Atomically download one file, skipping an existing verified destination."""
    if destination.is_file() and (
        expected_sha256 is None or sha256_file(destination) == expected_sha256
    ):
        return "verified_existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temporary:
                temporary_path = Path(temporary.name)
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return "downloaded"
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Download failed for {url}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_annotation(archive_path: Path, annotation_path: Path) -> str:
    if annotation_path.is_file():
        try:
            json.loads(annotation_path.read_text(encoding="utf-8"))
            return "verified_existing"
        except (OSError, json.JSONDecodeError):
            pass
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:xz") as archive:
        member = archive.getmember(ANNOTATION_MEMBER)
        if not member.isfile() or Path(member.name).name != ANNOTATION_MEMBER:
            raise ValueError("Annotation archive contains an unexpected member")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("Unable to read annotation JSON from archive")
        with tempfile.NamedTemporaryFile(delete=False, dir=annotation_path.parent) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(source.read())
    try:
        json.loads(temporary_path.read_text(encoding="utf-8"))
        os.replace(temporary_path, annotation_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "extracted"


def validate_annotation_coverage(annotation_path: Path, rows: Iterable[dict[str, str]]) -> None:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_names = {
        Path(image["file_name"]).as_posix().removeprefix("PKLot/UFPR04/")
        for image in annotation["images"]
    }
    missing = sorted(
        row["image_relpath"] for row in rows if row["image_relpath"] not in image_names
    )
    if missing:
        raise ValueError(f"Official annotation JSON is missing selected images: {missing[:3]}")


def download_subset(
    rows: list[dict[str, str]],
    raw_root: Path,
    image_base_url: str,
    annotation_url: str,
    annotation_sha256: str,
) -> dict[str, object]:
    raw_root = raw_root.resolve()
    image_statuses: dict[str, int] = {"downloaded": 0, "verified_existing": 0}
    for row in rows:
        relative = Path(row["image_relpath"])
        destination = safe_destination(raw_root, relative)
        url = f"{image_base_url.rstrip('/')}/{relative.as_posix()}"
        status = download_file(url, destination, row["image_sha256"])
        image_statuses[status] += 1

    annotation_dir = safe_destination(raw_root, Path("_annotations"))
    archive_path = annotation_dir / "ufpr04_spots.tar.xz"
    archive_status = download_file(annotation_url, archive_path, annotation_sha256)
    annotation_path = annotation_dir / ANNOTATION_MEMBER
    annotation_status = extract_annotation(archive_path, annotation_path)
    validate_annotation_coverage(annotation_path, rows)

    receipt = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "image_base_url": image_base_url,
        "annotation_archive_url": annotation_url,
        "annotation_archive_sha256": annotation_sha256,
        "images": {"count": len(rows), **image_statuses},
        "annotation_archive": {
            "path": str(archive_path.relative_to(raw_root)),
            "status": archive_status,
        },
        "annotation_json": {
            "path": str(annotation_path.relative_to(raw_root)),
            "sha256": sha256_file(annotation_path),
            "status": annotation_status,
        },
    }
    receipt_path = safe_destination(raw_root, Path("_receipts/ufpr04_subset_v1.download.json"))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.csv"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/PKLot/UFPR04_selected_v1"))
    parser.add_argument("--image-base-url", default=IMAGE_BASE_URL)
    parser.add_argument("--annotation-url", default=ANNOTATION_ARCHIVE_URL)
    parser.add_argument("--annotation-sha256", default=ANNOTATION_ARCHIVE_SHA256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest)
    receipt = download_subset(
        rows,
        validate_raw_root(args.raw_root),
        args.image_base_url,
        args.annotation_url,
        args.annotation_sha256,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
