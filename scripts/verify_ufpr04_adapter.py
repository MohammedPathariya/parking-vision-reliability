#!/usr/bin/env python3
"""Audit the official UFPR04 JSON labels used by the parking-space adapter."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from parking_vision_reliability.data.pklot import load_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_splits_v1.csv"),
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/raw/PKLot/UFPR04_selected_v1/_annotations/ufpr04_spots.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frames = load_frames(args.annotation, {row["image_relpath"] for row in rows})
    status_counts = Counter(
        space.ground_truth for frame in frames.values() for space in frame.spaces
    )
    legacy_mismatches = sum(
        len(frames[row["image_relpath"]].spaces) != int(row["labeled_space_count"])
        for row in rows
    )
    summary = {
        "frames": len(frames),
        "spaces": sum(len(frame.spaces) for frame in frames.values()),
        "ground_truth": dict(sorted(status_counts.items())),
        "legacy_manifest_label_count_mismatches": legacy_mismatches,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
