"""Torchvision COCO detector adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssdlite320_mobilenet_v3_large,
)

from parking_vision_reliability.occupancy.assignment import Detection

DetectorFactory = Callable[..., torch.nn.Module]


@dataclass(frozen=True)
class TorchvisionModelSpec:
    model_id: str
    factory: DetectorFactory
    weights: object


MODEL_SPECS = {
    "ssdlite320_mobilenet_v3_large": TorchvisionModelSpec(
        model_id="ssdlite320_mobilenet_v3_large",
        factory=ssdlite320_mobilenet_v3_large,
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "fasterrcnn_mobilenet_v3_large_fpn": TorchvisionModelSpec(
        model_id="fasterrcnn_mobilenet_v3_large_fpn",
        factory=fasterrcnn_mobilenet_v3_large_fpn,
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    ),
}


@dataclass
class TorchvisionDetector:
    """A common adapter around one official COCO-pretrained Torchvision detector."""

    model_id: str
    model: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    categories: list[str]
    device: torch.device

    def detect(self, image: Image.Image) -> list[Detection]:
        image_tensor = self.preprocess(image).to(self.device)
        with torch.inference_mode():
            output = self.model([image_tensor])[0]
        return normalize_output(output, self.categories)


def normalize_output(output: dict[str, torch.Tensor], categories: list[str]) -> list[Detection]:
    """Map Torchvision tensors to project-level COCO-category detections."""
    detections: list[Detection] = []
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        category_index = int(label.item())
        if not 0 <= category_index < len(categories):
            raise ValueError(f"Torchvision returned unknown COCO category index: {category_index}")
        xmin, ymin, xmax, ymax = (float(value) for value in box.tolist())
        detections.append(
            Detection(xmin, ymin, xmax, ymax, float(score.item()), categories[category_index])
        )
    return detections


def load_detector(model_id: str, device_name: str = "cpu") -> TorchvisionDetector:
    """Load one official pretrained Torchvision detector on the requested device."""
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unsupported detector: {model_id}")
    if device_name != "cpu":
        raise ValueError("Phase 1 smoke tests are CPU-only")
    spec = MODEL_SPECS[model_id]
    weights = spec.weights
    categories = list(weights.meta["categories"])
    device = torch.device(device_name)
    model = spec.factory(weights=weights).to(device).eval()
    return TorchvisionDetector(
        model_id=spec.model_id,
        model=model,
        preprocess=weights.transforms(),
        categories=categories,
        device=device,
    )
