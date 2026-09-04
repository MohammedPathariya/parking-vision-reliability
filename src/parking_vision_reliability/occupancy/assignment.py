"""Explainable vehicle-box to parking-space occupancy rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from parking_vision_reliability.data.pklot import ParkingSpace, Point

AssignmentMethod = Literal["center", "intersection_over_space"]
VEHICLE_CATEGORIES = frozenset({"car", "motorcycle", "bus", "truck"})


@dataclass(frozen=True)
class Detection:
    """A model-normalized vehicle detection in original-image coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float
    category: str

    def __post_init__(self) -> None:
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("Detection box must have positive area")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection score must be in [0, 1]")


@dataclass(frozen=True)
class SpaceDecision:
    annotation_id: int
    status: Literal["occupied", "available"]
    detector_confidence: float | None
    overlap_ratio: float


def polygon_area(polygon: tuple[Point, ...]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
        )
    ) / 2.0


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def clip_polygon_to_box(polygon: tuple[Point, ...], detection: Detection) -> tuple[Point, ...]:
    def at_x(x: float, start: Point, end: Point) -> Point:
        return x, start[1] + (end[1] - start[1]) * (x - start[0]) / (end[0] - start[0])

    def at_y(y: float, start: Point, end: Point) -> Point:
        return start[0] + (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]), y

    boundaries = (
        (
            lambda point: point[0] >= detection.xmin,
            lambda start, end: at_x(detection.xmin, start, end),
        ),
        (
            lambda point: point[0] <= detection.xmax,
            lambda start, end: at_x(detection.xmax, start, end),
        ),
        (
            lambda point: point[1] >= detection.ymin,
            lambda start, end: at_y(detection.ymin, start, end),
        ),
        (
            lambda point: point[1] <= detection.ymax,
            lambda start, end: at_y(detection.ymax, start, end),
        ),
    )
    clipped = list(polygon)
    for is_inside, intersection in boundaries:
        output: list[Point] = []
        for start, end in zip(clipped, clipped[1:] + clipped[:1]):
            start_inside, end_inside = is_inside(start), is_inside(end)
            if start_inside and end_inside:
                output.append(end)
            elif start_inside:
                output.append(intersection(start, end))
            elif end_inside:
                output.extend((intersection(start, end), end))
        clipped = output
        if not clipped:
            break
    return tuple(clipped)


def intersection_over_space(space: ParkingSpace, detection: Detection) -> float:
    area = polygon_area(space.polygon)
    if area == 0:
        raise ValueError(f"Parking-space polygon has zero area: {space.annotation_id}")
    return polygon_area(clip_polygon_to_box(space.polygon, detection)) / area


def assign_occupancy(
    spaces: tuple[ParkingSpace, ...],
    detections: list[Detection],
    method: AssignmentMethod,
    minimum_detection_score: float = 0.0,
    minimum_overlap_ratio: float = 0.5,
) -> list[SpaceDecision]:
    """Convert accepted vehicle detections into one occupied or available decision per space."""
    if method not in {"center", "intersection_over_space"}:
        raise ValueError(f"Unsupported assignment method: {method}")
    if not 0.0 <= minimum_detection_score <= 1.0:
        raise ValueError("minimum_detection_score must be in [0, 1]")
    if not 0.0 <= minimum_overlap_ratio <= 1.0:
        raise ValueError("minimum_overlap_ratio must be in [0, 1]")
    accepted = [
        detection
        for detection in detections
        if detection.category.lower() in VEHICLE_CATEGORIES
        and detection.score >= minimum_detection_score
    ]

    decisions: list[SpaceDecision] = []
    for space in spaces:
        candidates: list[tuple[Detection, float]] = []
        for detection in accepted:
            overlap = intersection_over_space(space, detection)
            if method == "center":
                center = (
                    (detection.xmin + detection.xmax) / 2,
                    (detection.ymin + detection.ymax) / 2,
                )
                if point_in_polygon(center, space.polygon):
                    candidates.append((detection, overlap))
            elif overlap >= minimum_overlap_ratio:
                candidates.append((detection, overlap))
        if candidates:
            detection, overlap = max(candidates, key=lambda item: item[0].score)
            decisions.append(
                SpaceDecision(space.annotation_id, "occupied", detection.score, overlap)
            )
        else:
            decisions.append(SpaceDecision(space.annotation_id, "available", None, 0.0))
    return decisions
