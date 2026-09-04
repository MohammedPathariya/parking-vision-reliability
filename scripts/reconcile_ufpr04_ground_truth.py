#!/usr/bin/env python3
"""Create a tracked MetaPKLot ground-truth manifest for the Phase 1 image splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from parking_vision_reliability.data.pklot import FrameParkingAnnotations, load_frames
from scripts.inventory_pklot import sha256_file, write_csv, write_json

GROUND_TRUTH_FIELDS = (
    "image_relpath",
    "split",
    "weather",
    "acquisition_date",
    "capture_time",
    "source_labeled_space_count",
    "source_occupied_count",
    "source_vacant_count",
    "source_unlabeled_space_count",
    "official_space_count",
    "official_occupied_count",
    "official_available_count",
)


def build_rows(
    split_rows: list[dict[str, str]], frames: dict[str, FrameParkingAnnotations]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in split_rows:
        frame = frames[source["image_relpath"]]
        official_counts = Counter(space.ground_truth for space in frame.spaces)
        rows.append(
            {
                "image_relpath": source["image_relpath"],
                "split": source["split"],
                "weather": source["weather"],
                "acquisition_date": source["acquisition_date"],
                "capture_time": source["capture_time"],
                "source_labeled_space_count": source["labeled_space_count"],
                "source_occupied_count": source["occupied_count"],
                "source_vacant_count": source["vacant_count"],
                "source_unlabeled_space_count": source["unlabeled_space_count"],
                "official_space_count": str(len(frame.spaces)),
                "official_occupied_count": str(official_counts["occupied"]),
                "official_available_count": str(official_counts["available"]),
            }
        )
    return rows


def build_metadata(
    rows: list[dict[str, str]], split_manifest: Path, annotation: Path, output: Path
) -> dict[str, object]:
    per_split: dict[str, dict[str, object]] = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        official_occupied = sum(int(row["official_occupied_count"]) for row in split_rows)
        official_spaces = sum(int(row["official_space_count"]) for row in split_rows)
        per_split[split] = {
            "images": len(split_rows),
            "official_occupied_spaces": official_occupied,
            "official_available_spaces": sum(
                int(row["official_available_count"]) for row in split_rows
            ),
            "official_occupancy_rate": round(official_occupied / official_spaces, 6),
        }
    source_space_count = sum(int(row["source_labeled_space_count"]) for row in rows)
    official_space_count = sum(int(row["official_space_count"]) for row in rows)
    return {
        "algorithm_version": "ufpr04-metapklot-ground-truth-v1",
        "dataset": "PKLot/UFPR04",
        "ground_truth_source": "MetaPKLot original UFPR04 spots JSON",
        "split_manifest_sha256": sha256_file(split_manifest),
        "annotation_sha256": sha256_file(annotation),
        "ground_truth_manifest_sha256": sha256_file(output),
        "reconciliation": {
            "source_manifest_label_count": source_space_count,
            "official_json_space_count": official_space_count,
            "frames_with_count_difference": sum(
                int(row["source_labeled_space_count"]) != int(row["official_space_count"])
                for row in rows
            ),
            "policy": (
                "Use official JSON labels and polygons for all detector assignment and benchmark "
                "metrics. Preserve XML-derived counts only as historical sampling provenance."
            ),
        },
        "splits": per_split,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_splits_v1.csv"),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/raw/PKLot/UFPR04_selected_v1/_annotations/ufpr04_spots.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_metapklot_ground_truth_v1.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path(
            "data/manifests/pklot_ufpr04_phase1_metapklot_ground_truth_v1.metadata.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.split_manifest.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    frames = load_frames(args.annotation, {row["image_relpath"] for row in split_rows})
    rows = build_rows(split_rows, frames)
    write_csv(args.output, rows, GROUND_TRUTH_FIELDS)
    metadata = build_metadata(rows, args.split_manifest, args.annotation, args.output)
    write_json(args.metadata_output, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
