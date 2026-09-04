#!/usr/bin/env python3
"""Audit the official UFPR04 JSON labels used by the parking-space adapter."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from parking_vision_reliability.data.pklot import load_frames
from scripts.reconcile_ufpr04_ground_truth import build_rows


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
    parser.add_argument(
        "--ground-truth-manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_metapklot_ground_truth_v1.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frames = load_frames(args.annotation, {row["image_relpath"] for row in rows})
    expected_rows = build_rows(rows, frames)
    errors: list[str] = []
    if not args.ground_truth_manifest.is_file():
        errors.append(f"missing ground-truth manifest: {args.ground_truth_manifest}")
    else:
        with args.ground_truth_manifest.open(newline="", encoding="utf-8") as handle:
            actual_rows = list(csv.DictReader(handle))
        if actual_rows != expected_rows:
            errors.append("ground-truth manifest does not match official JSON annotations")
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
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
