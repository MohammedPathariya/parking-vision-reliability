from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

from PIL import Image

from scripts.inventory_pklot import inventory_image
from scripts.select_ufpr04_subset import select_subset, validate_selection


def annotation_xml() -> str:
    return """<?xml version="1.0"?>
<parking id="ufpr04">
  <space id="1" occupied="1">
    <rotatedRect><center x="20" y="20"/><size w="10" h="20"/><angle d="0"/></rotatedRect>
    <contour>
      <point x="10" y="10"/><point x="30" y="10"/>
      <point x="30" y="30"/><point x="10" y="30"/>
    </contour>
  </space>
  <space id="2" occupied="0">
    <rotatedRect><center x="50" y="20"/><size w="10" h="20"/><angle d="0"/></rotatedRect>
    <contour>
      <point x="40" y="10"/><point x="60" y="10"/>
      <point x="60" y="30"/><point x="40" y="30"/>
    </contour>
  </space>
</parking>
"""


def synthetic_inventory_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for weather_index, weather in enumerate(("Sunny", "Cloudy", "Rainy")):
        for date_offset in range(6):
            acquisition_date = date(2020, weather_index + 1, date_offset + 1).isoformat()
            start = datetime.fromisoformat(f"{acquisition_date}T08:00:00")
            for frame_index in range(20):
                timestamp = start + timedelta(minutes=15 * frame_index)
                stem = timestamp.strftime("%Y-%m-%d_%H_%M_%S")
                rows.append(
                    {
                        "image_relpath": f"{weather}/{acquisition_date}/{stem}.jpg",
                        "annotation_relpath": f"{weather}/{acquisition_date}/{stem}.xml",
                        "weather": weather,
                        "acquisition_date": acquisition_date,
                        "capture_time": timestamp.strftime("%H:%M:%S"),
                        "width": "1280",
                        "height": "720",
                        "parking_space_count": "2",
                        "labeled_space_count": "2",
                        "unlabeled_space_count": "0",
                        "occupied_count": str(frame_index % 2),
                        "vacant_count": str(2 - (frame_index % 2)),
                        "occupancy_ratio": f"{(frame_index % 2) / 2:.6f}",
                        "image_sha256": f"image-{weather}-{date_offset}-{frame_index}",
                        "annotation_sha256": f"xml-{weather}-{date_offset}-{frame_index}",
                        "validation_status": "valid",
                    }
                )
    return rows


class InventoryTests(unittest.TestCase):
    def test_valid_image_and_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "Sunny" / "2020-01-01"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "2020-01-01_08_00_00.jpg"
            Image.new("RGB", (1280, 720), "white").save(image_path, "JPEG")
            image_path.with_suffix(".xml").write_text(annotation_xml(), encoding="utf-8")

            row = inventory_image(image_path, root)

            self.assertEqual(row["validation_status"], "valid")
            self.assertEqual(row["parking_space_count"], "2")
            self.assertEqual(row["labeled_space_count"], "2")
            self.assertEqual(row["unlabeled_space_count"], "0")
            self.assertEqual(row["occupied_count"], "1")
            self.assertEqual(row["vacant_count"], "1")


class SelectionTests(unittest.TestCase):
    def test_selection_is_balanced_unique_and_reproducible(self) -> None:
        inventory = synthetic_inventory_rows()
        first = select_subset(inventory)
        second = select_subset(inventory)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 90)
        validate_selection(first, dates_per_weather=3, images_per_date=10)


if __name__ == "__main__":
    unittest.main()
