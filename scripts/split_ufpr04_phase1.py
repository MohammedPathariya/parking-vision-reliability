#!/usr/bin/env python3
"""Create date-disjoint, weather-balanced Phase 1 splits from the UFPR04 source pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.inventory_pklot import WEATHERS, sha256_file, write_csv, write_json
from scripts.select_ufpr04_subset import SUBSET_FIELDS

SPLIT_NAMES = ("smoke", "calibration", "evaluation")
SPLIT_ALGORITHM_VERSION = "ufpr04-phase1-date-disjoint-v1"


def date_group_stats(rows: list[dict[str, str]]) -> tuple[int, int]:
    occupied = sum(int(row["occupied_count"]) for row in rows)
    labeled = sum(int(row["labeled_space_count"]) for row in rows)
    if not labeled:
        raise ValueError("A date group has no labeled parking spaces")
    return occupied, labeled


def split_rows(rows: list[dict[str, str]], seed: int = 2_026_090_2) -> list[dict[str, str]]:
    """Assign whole acquisition-date groups to balanced, non-overlapping splits."""
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["weather"]][row["acquisition_date"]].append(row)

    dates_by_weather: dict[str, list[str]] = {}
    for weather in WEATHERS:
        dates = sorted(groups[weather])
        if len(dates) != len(SPLIT_NAMES):
            raise ValueError(
                f"{weather} has {len(dates)} selected dates; need exactly {len(SPLIT_NAMES)}"
            )
        group_sizes = {date: len(groups[weather][date]) for date in dates}
        if any(size != 10 for size in group_sizes.values()):
            raise ValueError(f"{weather} date groups must contain 10 frames: {group_sizes}")
        dates_by_weather[weather] = dates

    target_occupied, target_labeled = date_group_stats(rows)
    target_ratio = target_occupied / target_labeled
    weather_permutations = [
        list(itertools.permutations(dates_by_weather[weather])) for weather in WEATHERS
    ]

    def assignment_rank(assignments: tuple[tuple[str, ...], ...]) -> tuple[float, float, str]:
        ratios: list[float] = []
        for index in range(len(SPLIT_NAMES)):
            split_rows_for_index = [
                groups[weather][assignments[weather_index][index]]
                for weather_index, weather in enumerate(WEATHERS)
            ]
            occupied, labeled = date_group_stats(
                [row for date_rows in split_rows_for_index for row in date_rows]
            )
            ratios.append(occupied / labeled)
        serialized = "|".join(
            ":".join(assignment) for assignment in assignments
        )
        tie_breaker = hashlib.sha256(f"{seed}:{serialized}".encode()).hexdigest()
        deviations = [abs(ratio - target_ratio) for ratio in ratios]
        return max(deviations), sum(deviations), tie_breaker

    assignments = min(itertools.product(*weather_permutations), key=assignment_rank)
    split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    weather_order = {weather: index for index, weather in enumerate(WEATHERS)}
    output: list[dict[str, str]] = []
    for split_index, split_name in enumerate(SPLIT_NAMES):
        for weather_index, weather in enumerate(WEATHERS):
            date = assignments[weather_index][split_index]
            output.extend({**row, "split": split_name} for row in groups[weather][date])

    output.sort(
        key=lambda row: (
            split_order[row["split"]],
            weather_order[row["weather"]],
            row["acquisition_date"],
            row["capture_time"],
        )
    )
    validate_splits(output, rows)
    return output


def validate_splits(
    split_rows_output: list[dict[str, str]], source_rows: list[dict[str, str]]
) -> None:
    if len(split_rows_output) != len(source_rows):
        raise ValueError("Split rows do not cover the entire source pool")
    source_paths = {row["image_relpath"] for row in source_rows}
    output_paths = [row["image_relpath"] for row in split_rows_output]
    if len(output_paths) != len(set(output_paths)) or set(output_paths) != source_paths:
        raise ValueError("Split rows do not contain each source image exactly once")

    split_counts = Counter(row["split"] for row in split_rows_output)
    if set(split_counts) != set(SPLIT_NAMES) or any(
        split_counts[split] != 30 for split in SPLIT_NAMES
    ):
        raise ValueError(f"Expected 30 images per split, found {dict(split_counts)}")

    dates_to_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in split_rows_output:
        dates_to_splits[(row["weather"], row["acquisition_date"])].add(row["split"])
    if any(len(splits) != 1 for splits in dates_to_splits.values()):
        raise ValueError("An acquisition date appears in more than one split")

    for split in SPLIT_NAMES:
        split_weather_counts = Counter(
            row["weather"] for row in split_rows_output if row["split"] == split
        )
        if any(split_weather_counts[weather] != 10 for weather in WEATHERS):
            raise ValueError(f"{split} is not weather-balanced: {dict(split_weather_counts)}")


def read_source_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBSET_FIELDS:
            raise ValueError(f"Unexpected source manifest schema in {path}")
        rows = list(reader)
    if any(row["split"] != "unassigned" for row in rows):
        raise ValueError("Source manifest must contain only unassigned rows")
    return rows


def split_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split in SPLIT_NAMES:
        split_rows_for_name = [row for row in rows if row["split"] == split]
        occupied, labeled = date_group_stats(split_rows_for_name)
        summary[split] = {
            "images": len(split_rows_for_name),
            "dates_by_weather": {
                weather: sorted(
                    {
                        row["acquisition_date"]
                        for row in split_rows_for_name
                        if row["weather"] == weather
                    }
                )
                for weather in WEATHERS
            },
            "images_by_weather": {
                weather: sum(1 for row in split_rows_for_name if row["weather"] == weather)
                for weather in WEATHERS
            },
            "occupied_spaces": occupied,
            "labeled_spaces": labeled,
            "occupancy_rate": round(occupied / labeled, 6),
        }
    return summary


def build_metadata(
    rows: list[dict[str, str]], source_manifest: Path, output: Path, seed: int
) -> dict[str, object]:
    return {
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "dataset": "PKLot/UFPR04",
        "source_manifest_sha256": sha256_file(source_manifest),
        "split_manifest_sha256": sha256_file(output),
        "selection_seed": seed,
        "constraints": {
            "date_disjoint": True,
            "whole_acquisition_dates": True,
            "images_per_split": 30,
            "images_per_weather_per_split": 10,
            "split_names": list(SPLIT_NAMES),
        },
        "assignment_strategy": (
            "Enumerate all whole-date assignments and minimize the maximum deviation of a "
            "split occupancy rate from the 90-image source-pool occupancy rate; seed breaks ties."
        ),
        "splits": split_summary(rows),
        "limitations": [
            "Each split contains one acquisition date per weather condition.",
            "The source pool has only three dates per weather, so a 12/48/30 split would "
            "leak dates.",
            "The evaluation split is date-disjoint but not a future-only temporal split.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_splits_v1.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_splits_v1.metadata.json"),
    )
    parser.add_argument("--seed", type=int, default=2_026_090_2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_source_manifest(args.source_manifest)
    output_rows = split_rows(source_rows, seed=args.seed)
    write_csv(args.output, output_rows, SUBSET_FIELDS)
    for split in SPLIT_NAMES:
        write_csv(
            args.output.with_name(f"{args.output.stem}_{split}.csv"),
            [row for row in output_rows if row["split"] == split],
            SUBSET_FIELDS,
        )
    metadata = build_metadata(output_rows, args.source_manifest, args.output, args.seed)
    write_json(args.metadata_output, metadata)
    print(json.dumps(metadata["splits"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
