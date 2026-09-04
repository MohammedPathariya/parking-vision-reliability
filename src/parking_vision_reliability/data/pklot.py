"""Read MetaPKLot's UFPR04 parking-space annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Point = tuple[float, float]


@dataclass(frozen=True)
class ParkingSpace:
    """One parking-space polygon and its source occupancy label for a frame."""

    annotation_id: int
    polygon: tuple[Point, ...]
    ground_truth: str


@dataclass(frozen=True)
class FrameParkingAnnotations:
    """All labeled parking spaces for one UFPR04 image."""

    image_relpath: str
    width: int
    height: int
    spaces: tuple[ParkingSpace, ...]


def image_relpath(file_name: str) -> str:
    prefix = "PKLot/UFPR04/"
    if not file_name.startswith(prefix):
        raise ValueError(f"Expected a UFPR04 image path, got {file_name!r}")
    return file_name.removeprefix(prefix)


def polygon_from_segmentation(segmentation: object) -> tuple[Point, ...]:
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError("Parking-space annotation has no polygon segmentation")
    candidates = [part for part in segmentation if isinstance(part, list) and len(part) >= 6]
    if not candidates:
        raise ValueError("Parking-space annotation has no valid polygon segmentation")
    coordinates = max(candidates, key=len)
    if len(coordinates) % 2:
        raise ValueError("Parking-space polygon has an odd coordinate count")
    return tuple(
        (float(coordinates[index]), float(coordinates[index + 1]))
        for index in range(0, len(coordinates), 2)
    )


def load_frames(
    annotation_path: Path, selected_image_relpaths: set[str] | None = None
) -> dict[str, FrameParkingAnnotations]:
    """Load selected UFPR04 images into frame-local parking-space annotations."""
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in document["categories"]}
    expected_categories = {0: "empty", 1: "occupied"}
    if categories != expected_categories:
        raise ValueError(f"Unexpected UFPR04 parking categories: {categories}")

    images = {
        image["id"]: (image_relpath(image["file_name"]), int(image["width"]), int(image["height"]))
        for image in document["images"]
    }
    selected_ids = {
        image_id
        for image_id, (relative, _, _) in images.items()
        if selected_image_relpaths is None or relative in selected_image_relpaths
    }
    if selected_image_relpaths is not None:
        found = {images[image_id][0] for image_id in selected_ids}
        missing = sorted(selected_image_relpaths - found)
        if missing:
            raise ValueError(f"Selected images are missing from annotation JSON: {missing[:3]}")

    spaces_by_image: dict[int, list[ParkingSpace]] = {image_id: [] for image_id in selected_ids}
    for annotation in document["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in selected_ids:
            continue
        category_id = annotation["category_id"]
        if category_id not in categories:
            raise ValueError(f"Unknown parking category ID: {category_id}")
        spaces_by_image[image_id].append(
            ParkingSpace(
                annotation_id=int(annotation["id"]),
                polygon=polygon_from_segmentation(annotation["segmentation"]),
                ground_truth="occupied" if category_id == 1 else "available",
            )
        )

    frames: dict[str, FrameParkingAnnotations] = {}
    for image_id in selected_ids:
        relative, width, height = images[image_id]
        spaces = tuple(sorted(spaces_by_image[image_id], key=lambda space: space.annotation_id))
        if not spaces:
            raise ValueError(f"Selected image has no parking-space labels: {relative}")
        frames[relative] = FrameParkingAnnotations(relative, width, height, spaces)
    return frames
