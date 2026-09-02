#!/usr/bin/env python3
"""Select and materialize a deterministic weather-balanced UFPR04 subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

from scripts.inventory_pklot import INVENTORY_FIELDS, WEATHERS, sha256_file, write_csv, write_json

ALGORITHM_VERSION = "ufpr04-balanced-v1"
SOURCE_ARCHIVE_URL = "https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz"
SOURCE_ARCHIVE_SIZE_BYTES = 4_898_276_304
SOURCE_ARCHIVE_ETAG = "574da3cd-123f5c3d0"
SOURCE_ARCHIVE_SHA256 = "e89bbc1dc735298c478688d50c7a682fb3b0076a87b6634923132709f2d2fa9b"
SUBSET_FIELDS = (
    "subset_id",
    "split",
    "selection_seed",
    "date_bucket",
    *INVENTORY_FIELDS,
)


def parse_timestamp(row: dict[str, str]) -> datetime:
    return datetime.fromisoformat(f"{row['acquisition_date']}T{row['capture_time']}")


def spaced_candidates(
    rows: Iterable[dict[str, str]], minimum_gap_minutes: int
) -> list[dict[str, str]]:
    ordered = sorted(rows, key=parse_timestamp)
    selected: list[dict[str, str]] = []
    last_timestamp: datetime | None = None
    gap = timedelta(minutes=minimum_gap_minutes)
    for row in ordered:
        timestamp = parse_timestamp(row)
        if last_timestamp is None or timestamp - last_timestamp >= gap:
            selected.append(row)
            last_timestamp = timestamp
    return selected


def partition(items: list[dict[str, str]] | list[str], parts: int) -> list[list]:
    return [
        items[(index * len(items)) // parts : ((index + 1) * len(items)) // parts]
        for index in range(parts)
    ]


def select_subset(
    inventory_rows: list[dict[str, str]],
    seed: int = 2_026_090_2,
    dates_per_weather: int = 3,
    images_per_date: int = 10,
    minimum_gap_minutes: int = 15,
    subset_id: str = "ufpr04_balanced_v1",
) -> list[dict[str, str]]:
    if dates_per_weather <= 0 or images_per_date <= 0:
        raise ValueError("dates_per_weather and images_per_date must be positive")

    valid_rows = [row for row in inventory_rows if row["validation_status"] == "valid"]
    by_weather_date: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in valid_rows:
        by_weather_date[row["weather"]][row["acquisition_date"]].append(row)

    result: list[dict[str, str]] = []
    for weather in WEATHERS:
        eligible: dict[str, list[dict[str, str]]] = {}
        for acquisition_date, rows in by_weather_date[weather].items():
            candidates = spaced_candidates(rows, minimum_gap_minutes)
            if len(candidates) >= images_per_date:
                eligible[acquisition_date] = candidates

        eligible_dates = sorted(eligible)
        if len(eligible_dates) < dates_per_weather:
            raise ValueError(
                f"{weather} has {len(eligible_dates)} eligible dates; "
                f"need {dates_per_weather} with {images_per_date} frames at "
                f"least {minimum_gap_minutes} minutes apart"
            )

        date_buckets = partition(eligible_dates, dates_per_weather)
        if any(not bucket for bucket in date_buckets):
            raise ValueError(f"Unable to create {dates_per_weather} date buckets for {weather}")

        frames_by_date: dict[str, list[dict[str, str]]] = {}
        for acquisition_date, candidates in eligible.items():
            time_buckets = partition(candidates, images_per_date)
            if any(not bucket for bucket in time_buckets):
                raise ValueError(f"Insufficient time coverage for {weather}/{acquisition_date}")
            frames_by_date[acquisition_date] = [
                random.Random(
                    f"{seed}:{weather}:{acquisition_date}:frame:{time_bucket_index}"
                ).choice(time_bucket)
                for time_bucket_index, time_bucket in enumerate(time_buckets, start=1)
            ]

        weather_rows = [row for row in valid_rows if row["weather"] == weather]
        target_ratio = sum(int(row["occupied_count"]) for row in weather_rows) / sum(
            int(row["labeled_space_count"]) for row in weather_rows
        )

        def combination_rank(selected_dates: tuple[str, ...]) -> tuple[float, str]:
            frames = [row for day in selected_dates for row in frames_by_date[day]]
            selected_ratio = sum(int(row["occupied_count"]) for row in frames) / sum(
                int(row["labeled_space_count"]) for row in frames
            )
            tie_breaker = hashlib.sha256(
                f"{seed}:{weather}:{':'.join(selected_dates)}".encode()
            ).hexdigest()
            return abs(selected_ratio - target_ratio), tie_breaker

        selected_dates = min(itertools.product(*date_buckets), key=combination_rank)

        for bucket_index, selected_date in enumerate(selected_dates, start=1):
            for source_row in frames_by_date[selected_date]:
                result.append(
                    {
                        "subset_id": subset_id,
                        "split": "evaluation",
                        "selection_seed": str(seed),
                        "date_bucket": str(bucket_index),
                        **source_row,
                    }
                )

    weather_order = {weather: index for index, weather in enumerate(WEATHERS)}
    return sorted(
        result,
        key=lambda row: (
            weather_order[row["weather"]],
            row["acquisition_date"],
            row["capture_time"],
        ),
    )


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
            raise ValueError(f"Unexpected inventory schema in {path}")
        return list(reader)


def materialize_subset(
    rows: Iterable[dict[str, str]], dataset_root: Path, subset_root: Path
) -> None:
    for row in rows:
        for field in ("image_relpath", "annotation_relpath"):
            relative = Path(row[field])
            source = dataset_root / relative
            destination = subset_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def build_metadata(
    rows: list[dict[str, str]],
    inventory_rows: list[dict[str, str]],
    inventory_path: Path,
    manifest_path: Path,
    seed: int,
    dates_per_weather: int,
    images_per_date: int,
    minimum_gap_minutes: int,
    subset_id: str,
) -> dict[str, object]:
    weather_counts = Counter(row["weather"] for row in rows)
    occupied_counts = Counter()
    vacant_counts = Counter()
    unlabeled_counts = Counter()
    inventory_occupied_counts = Counter()
    inventory_labeled_counts = Counter()
    dates: dict[str, list[str]] = {}
    for weather in WEATHERS:
        weather_rows = [row for row in rows if row["weather"] == weather]
        occupied_counts[weather] = sum(int(row["occupied_count"]) for row in weather_rows)
        vacant_counts[weather] = sum(int(row["vacant_count"]) for row in weather_rows)
        unlabeled_counts[weather] = sum(int(row["unlabeled_space_count"]) for row in weather_rows)
        dates[weather] = sorted({row["acquisition_date"] for row in weather_rows})
        inventory_weather_rows = [row for row in inventory_rows if row["weather"] == weather]
        inventory_occupied_counts[weather] = sum(
            int(row["occupied_count"]) for row in inventory_weather_rows
        )
        inventory_labeled_counts[weather] = sum(
            int(row["labeled_space_count"]) for row in inventory_weather_rows
        )

    return {
        "algorithm_version": ALGORITHM_VERSION,
        "dataset": "PKLot/UFPR04",
        "subset_id": subset_id,
        "split": "evaluation",
        "selection": {
            "seed": seed,
            "dates_per_weather": dates_per_weather,
            "images_per_date": images_per_date,
            "minimum_gap_minutes": minimum_gap_minutes,
            "date_selection": (
                "one date per chronological bucket, choosing the combination closest to the "
                "full-weather occupancy rate; seed breaks ties"
            ),
            "frame_selection": (
                "one seeded frame from each chronological time bucket after gap filtering"
            ),
        },
        "actual": {
            "total_images": len(rows),
            "images_by_weather": {weather: weather_counts[weather] for weather in WEATHERS},
            "dates_by_weather": dates,
            "occupied_spaces_by_weather": {
                weather: occupied_counts[weather] for weather in WEATHERS
            },
            "vacant_spaces_by_weather": {weather: vacant_counts[weather] for weather in WEATHERS},
            "unlabeled_spaces_by_weather": {
                weather: unlabeled_counts[weather] for weather in WEATHERS
            },
            "occupancy_rate_by_weather": {
                weather: round(
                    occupied_counts[weather] / (occupied_counts[weather] + vacant_counts[weather]),
                    6,
                )
                for weather in WEATHERS
            },
            "inventory_occupancy_rate_by_weather": {
                weather: round(
                    inventory_occupied_counts[weather] / inventory_labeled_counts[weather], 6
                )
                for weather in WEATHERS
            },
        },
        "source": {
            "archive_url": SOURCE_ARCHIVE_URL,
            "archive_size_bytes": SOURCE_ARCHIVE_SIZE_BYTES,
            "archive_etag": SOURCE_ARCHIVE_ETAG,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "inventory_sha256": sha256_file(inventory_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "license": {
            "spdx": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "citation": {
            "authors": (
                "P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., "
                "A. L. Koerich"
            ),
            "title": "PKLot - A robust dataset for parking lot classification",
            "venue": "Expert Systems with Applications 42(11), 4937-4949",
            "year": 2015,
            "doi": "10.1016/j.eswa.2015.02.009",
        },
        "limitations": [
            "Weather-balanced full frames are not class-balanced parking-space crops.",
            "PKLot is time-lapse imagery, not continuous video.",
            "All selected images come from one fixed UFPR04 camera view.",
        ],
    }


def validate_selection(
    rows: list[dict[str, str]], dates_per_weather: int, images_per_date: int
) -> None:
    expected_per_weather = dates_per_weather * images_per_date
    counts = Counter(row["weather"] for row in rows)
    if any(counts[weather] != expected_per_weather for weather in WEATHERS):
        raise ValueError(f"Weather balance check failed: {dict(counts)}")
    paths = [row["image_relpath"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths in selection")
    image_hashes = [row["image_sha256"] for row in rows]
    if len(image_hashes) != len(set(image_hashes)):
        raise ValueError("Duplicate image content in selection")
    for weather in WEATHERS:
        dates = {row["acquisition_date"] for row in rows if row["weather"] == weather}
        if len(dates) != dates_per_weather:
            raise ValueError(f"Date diversity check failed for {weather}: {sorted(dates)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_inventory.csv"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/PKLot/UFPR04"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.metadata.json"),
    )
    parser.add_argument(
        "--materialize-root",
        type=Path,
        default=Path("data/processed/ufpr04_balanced_v1"),
    )
    parser.add_argument("--seed", type=int, default=2_026_090_2)
    parser.add_argument("--dates-per-weather", type=int, default=3)
    parser.add_argument("--images-per-date", type=int, default=10)
    parser.add_argument("--minimum-gap-minutes", type=int, default=15)
    parser.add_argument("--subset-id", default="ufpr04_balanced_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory_rows = read_inventory(args.inventory)
    selected = select_subset(
        inventory_rows,
        seed=args.seed,
        dates_per_weather=args.dates_per_weather,
        images_per_date=args.images_per_date,
        minimum_gap_minutes=args.minimum_gap_minutes,
        subset_id=args.subset_id,
    )
    validate_selection(selected, args.dates_per_weather, args.images_per_date)
    write_csv(args.output, selected, SUBSET_FIELDS)
    materialize_subset(selected, args.dataset_root, args.materialize_root)
    metadata = build_metadata(
        selected,
        inventory_rows,
        args.inventory,
        args.output,
        args.seed,
        args.dates_per_weather,
        args.images_per_date,
        args.minimum_gap_minutes,
        args.subset_id,
    )
    write_json(args.metadata_output, metadata)
    print(json.dumps(metadata["actual"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
