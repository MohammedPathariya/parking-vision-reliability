"""Model-independent detector contract."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from parking_vision_reliability.occupancy.assignment import Detection


class VehicleDetector(Protocol):
    """A pretrained detector that emits original-image vehicle detections."""

    model_id: str

    def detect(self, image: Image.Image) -> list[Detection]:
        """Return COCO-labeled detections in the input image's pixel coordinates."""
