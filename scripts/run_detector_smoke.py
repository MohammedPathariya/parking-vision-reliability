#!/usr/bin/env python3
"""Run both pretrained Torchvision detectors on the frozen UFPR04 smoke split."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw

from parking_vision_reliability.data.pklot import load_frames
from parking_vision_reliability.detection.torchvision import MODEL_SPECS, load_detector
from parking_vision_reliability.occupancy.assignment import (
    VEHICLE_CATEGORIES,
    Detection,
    assign_occupancy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_phase1_splits_v1_smoke.csv"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/PKLot/UFPR04_selected_v1"))
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/raw/PKLot/UFPR04_selected_v1/_annotations/ufpr04_spots.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/detector_smoke"))
    parser.add_argument("--minimum-detection-score", type=float, default=0.1)
    parser.add_argument("--minimum-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--warmup-images", type=int, default=1)
    return parser.parse_args()


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * 0.95)]


def draw_preview(
    image: Image.Image,
    detections: list[Detection],
    polygons: list[tuple[tuple[float, float], ...]],
) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for polygon in polygons:
        draw.line([*polygon, polygon[0]], fill="cyan", width=2)
    for detection in detections:
        draw.rectangle(
            (detection.xmin, detection.ymin, detection.xmax, detection.ymax), outline="red", width=2
        )
        draw.text(
            (detection.xmin, detection.ymin),
            f"{detection.category} {detection.score:.2f}",
            fill="red",
        )
    return preview


def run_model(
    model_id: str,
    rows: list[dict[str, str]],
    raw_root: Path,
    frames: dict,
    output_dir: Path,
    minimum_detection_score: float,
    minimum_overlap_ratio: float,
    warmup_images: int,
) -> dict[str, object]:
    detector = load_detector(model_id)
    for row in rows[:warmup_images]:
        with Image.open(raw_root / row["image_relpath"]) as image:
            detector.detect(image.convert("RGB"))

    timings_ms: list[float] = []
    image_results: list[dict[str, object]] = []
    preview_dir = output_dir / model_id / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_path = raw_root / row["image_relpath"]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        started = time.perf_counter()
        detections = detector.detect(image)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        timings_ms.append(elapsed_ms)
        vehicles = [
            detection
            for detection in detections
            if detection.category.lower() in VEHICLE_CATEGORIES
            and detection.score >= minimum_detection_score
        ]
        frame = frames[row["image_relpath"]]
        decisions = {
            method: assign_occupancy(
                frame.spaces,
                detections,
                method,
                minimum_detection_score,
                minimum_overlap_ratio,
            )
            for method in ("center", "intersection_over_space")
        }
        preview_path = preview_dir / Path(row["image_relpath"]).name
        draw_preview(image, vehicles, [space.polygon for space in frame.spaces]).save(preview_path)
        image_results.append(
            {
                "image_relpath": row["image_relpath"],
                "weather": row["weather"],
                "latency_ms": round(elapsed_ms, 3),
                "all_detections": len(detections),
                "accepted_vehicle_detections": len(vehicles),
                "center_occupied_spaces": sum(
                    decision.status == "occupied" for decision in decisions["center"]
                ),
                "overlap_occupied_spaces": sum(
                    decision.status == "occupied"
                    for decision in decisions["intersection_over_space"]
                ),
            }
        )
    return {
        "model_id": model_id,
        "images": len(rows),
        "latency_ms": {
            "median": round(statistics.median(timings_ms), 3),
            "p95": round(percentile_95(timings_ms), 3),
        },
        "image_results": image_results,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.minimum_detection_score <= 1.0:
        raise ValueError("--minimum-detection-score must be in [0, 1]")
    if not 0.0 <= args.minimum_overlap_ratio <= 1.0:
        raise ValueError("--minimum-overlap-ratio must be in [0, 1]")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30 or {row["split"] for row in rows} != {"smoke"}:
        raise ValueError("Smoke manifest must contain exactly 30 smoke-split images")
    frames = load_frames(args.annotation, {row["image_relpath"] for row in rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol": {
            "split": "smoke",
            "device": "cpu",
            "warmup_images_per_model": args.warmup_images,
            "minimum_detection_score": args.minimum_detection_score,
            "minimum_overlap_ratio": args.minimum_overlap_ratio,
            "note": "Smoke-only thresholds are provisional and are not calibration results.",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "models": [
            run_model(
                model_id,
                rows,
                args.raw_root,
                frames,
                args.output_dir,
                args.minimum_detection_score,
                args.minimum_overlap_ratio,
                args.warmup_images,
            )
            for model_id in MODEL_SPECS
        ],
    }
    result_path = args.output_dir / "smoke_results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
