#!/usr/bin/env python3
"""Inventory and validate original PKLot UFPR04 JPEG/XML pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

WEATHERS = ("Sunny", "Cloudy", "Rainy")
EXPECTED_IMAGE_COUNTS = {"Sunny": 2_098, "Cloudy": 1_408, "Rainy": 285}
EXPECTED_LABELED_SPACE_COUNT = 105_845
EXPECTED_DIMENSIONS = (1280, 720)
TIMESTAMP_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})$"
)
INVENTORY_FIELDS = (
    "image_relpath",
    "annotation_relpath",
    "weather",
    "acquisition_date",
    "capture_time",
    "width",
    "height",
    "parking_space_count",
    "labeled_space_count",
    "unlabeled_space_count",
    "occupied_count",
    "vacant_count",
    "occupancy_ratio",
    "image_sha256",
    "annotation_sha256",
    "validation_status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions without requiring an imaging dependency."""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("missing JPEG start marker")

        while True:
            byte = handle.read(1)
            if not byte:
                raise ValueError("JPEG dimensions not found")
            if byte != b"\xff":
                continue

            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                raise ValueError("truncated JPEG marker")

            marker_value = marker[0]
            if marker_value in {0xD8, 0xD9} or 0xD0 <= marker_value <= 0xD7:
                continue

            raw_length = handle.read(2)
            if len(raw_length) != 2:
                raise ValueError("truncated JPEG segment")
            segment_length = struct.unpack(">H", raw_length)[0]
            if segment_length < 2:
                raise ValueError("invalid JPEG segment length")

            if marker_value in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = handle.read(5)
                if len(payload) != 5:
                    raise ValueError("truncated JPEG size segment")
                height, width = struct.unpack(">HH", payload[1:])
                return width, height

            handle.seek(segment_length - 2, os.SEEK_CUR)


def parse_annotation(
    path: Path, width: int, height: int
) -> tuple[int, int, int, int, int, list[str]]:
    issues: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return 0, 0, 0, 0, 0, [f"annotation_parse_error:{type(exc).__name__}"]

    if root.tag != "parking":
        issues.append("unexpected_annotation_root")
    if root.get("id", "").lower() != "ufpr04":
        issues.append("unexpected_parking_id")

    spaces = root.findall("space")
    occupied = 0
    vacant = 0
    unlabeled = 0
    seen_ids: set[str] = set()

    for space in spaces:
        space_id = space.get("id")
        if not space_id or space_id in seen_ids:
            issues.append("missing_or_duplicate_space_id")
        else:
            seen_ids.add(space_id)

        label = space.get("occupied")
        if label == "1":
            occupied += 1
        elif label == "0":
            vacant += 1
        elif label is None:
            unlabeled += 1
        else:
            issues.append("invalid_occupancy_label")

        contour = space.find("contour")
        points = (
            [child for child in contour if child.tag.lower() == "point"]
            if contour is not None
            else []
        )
        if len(points) < 3:
            issues.append("invalid_contour")
            continue
        for point in points:
            try:
                x = int(point.attrib["x"])
                y = int(point.attrib["y"])
            except (KeyError, ValueError):
                issues.append("invalid_contour_coordinate")
                continue
            if not (0 <= x < width and 0 <= y < height):
                issues.append("contour_out_of_bounds")

    if not spaces:
        issues.append("no_parking_spaces")

    return (
        len(spaces),
        occupied + vacant,
        unlabeled,
        occupied,
        vacant,
        sorted(set(issues)),
    )


def inventory_image(image_path: Path, dataset_root: Path) -> dict[str, str]:
    issues: list[str] = []
    relative = image_path.relative_to(dataset_root)
    parts = relative.parts
    weather = parts[0] if len(parts) >= 1 else ""
    acquisition_date = parts[1] if len(parts) >= 2 else ""

    match = TIMESTAMP_PATTERN.fullmatch(image_path.stem)
    capture_time = ""
    if match:
        filename_date = match.group("date")
        capture_time = ":".join((match.group("hour"), match.group("minute"), match.group("second")))
        if acquisition_date != filename_date:
            issues.append("date_path_filename_mismatch")
    else:
        issues.append("invalid_filename_timestamp")

    if weather not in WEATHERS:
        issues.append("unexpected_weather")

    annotation_path = image_path.with_suffix(".xml")
    width = 0
    height = 0
    try:
        width, height = jpeg_dimensions(image_path)
        if (width, height) != EXPECTED_DIMENSIONS:
            issues.append("unexpected_dimensions")
    except (OSError, ValueError) as exc:
        issues.append(f"image_parse_error:{type(exc).__name__}")

    parking_space_count = labeled_space_count = unlabeled_space_count = 0
    occupied_count = vacant_count = 0
    annotation_hash = ""
    if annotation_path.is_file():
        annotation_hash = sha256_file(annotation_path)
        if width and height:
            (
                parking_space_count,
                labeled_space_count,
                unlabeled_space_count,
                occupied_count,
                vacant_count,
                annotation_issues,
            ) = parse_annotation(annotation_path, width, height)
            issues.extend(annotation_issues)
    else:
        issues.append("missing_annotation")

    occupancy_ratio = f"{occupied_count / labeled_space_count:.6f}" if labeled_space_count else ""
    return {
        "image_relpath": relative.as_posix(),
        "annotation_relpath": (
            annotation_path.relative_to(dataset_root).as_posix()
            if annotation_path.is_file()
            else ""
        ),
        "weather": weather,
        "acquisition_date": acquisition_date,
        "capture_time": capture_time,
        "width": str(width),
        "height": str(height),
        "parking_space_count": str(parking_space_count),
        "labeled_space_count": str(labeled_space_count),
        "unlabeled_space_count": str(unlabeled_space_count),
        "occupied_count": str(occupied_count),
        "vacant_count": str(vacant_count),
        "occupancy_ratio": occupancy_ratio,
        "image_sha256": sha256_file(image_path),
        "annotation_sha256": annotation_hash,
        "validation_status": "valid" if not issues else ";".join(sorted(set(issues))),
    }


def build_inventory(dataset_root: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"UFPR04 root not found: {dataset_root}")

    image_paths = sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() == ".jpg"
    )
    rows = [inventory_image(path, dataset_root) for path in image_paths]

    image_stems = {path.with_suffix("") for path in image_paths}
    annotation_paths = {
        path.with_suffix("")
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xml"
    }
    weather_counts = Counter(row["weather"] for row in rows)
    status_counts = Counter(row["validation_status"] for row in rows)
    image_hash_counts = Counter(row["image_sha256"] for row in rows)
    annotation_hash_counts = Counter(row["annotation_sha256"] for row in rows)
    labeled_space_count = sum(int(row["labeled_space_count"]) for row in rows)
    unlabeled_space_count = sum(int(row["unlabeled_space_count"]) for row in rows)
    summary: dict[str, object] = {
        "dataset": "PKLot/UFPR04",
        "expected_image_counts": EXPECTED_IMAGE_COUNTS,
        "actual_image_counts": {weather: weather_counts[weather] for weather in WEATHERS},
        "total_images": len(rows),
        "valid_images": status_counts["valid"],
        "invalid_images": len(rows) - status_counts["valid"],
        "labeled_space_count": labeled_space_count,
        "expected_labeled_space_count": EXPECTED_LABELED_SPACE_COUNT,
        "unlabeled_space_count": unlabeled_space_count,
        "orphan_annotations": len(annotation_paths - image_stems),
        "duplicate_image_hash_groups": sum(
            count > 1 for image_hash, count in image_hash_counts.items() if image_hash
        ),
        "duplicate_annotation_hash_groups": sum(
            count > 1
            for annotation_hash, count in annotation_hash_counts.items()
            if annotation_hash
        ),
        "validation_status_counts": dict(sorted(status_counts.items())),
    }
    return rows, summary


def write_csv(path: Path, rows: Iterable[dict[str, str]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/PKLot/UFPR04"),
        help="Path to the extracted UFPR04 directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_inventory.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_inventory.summary.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless all rows are valid and published weather counts match",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, summary = build_inventory(args.dataset_root)
    write_csv(args.output, rows, INVENTORY_FIELDS)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    counts_match = summary["actual_image_counts"] == EXPECTED_IMAGE_COUNTS
    labels_match = summary["labeled_space_count"] == EXPECTED_LABELED_SPACE_COUNT
    clean = summary["invalid_images"] == 0 and summary["orphan_annotations"] == 0
    if args.strict and not (counts_match and labels_match and clean):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
