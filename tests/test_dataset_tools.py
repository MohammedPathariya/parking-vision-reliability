from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from parking_vision_reliability.data.pklot import load_frames
from parking_vision_reliability.occupancy.assignment import Detection, assign_occupancy
from scripts.download_ufpr04_subset import download_subset, safe_destination, validate_raw_root
from scripts.inventory_pklot import inventory_image
from scripts.select_ufpr04_subset import select_subset, validate_selection
from scripts.split_ufpr04_phase1 import split_rows, validate_splits


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

    def test_phase1_splits_are_date_disjoint_and_reproducible(self) -> None:
        source = []
        for weather in ("Sunny", "Cloudy", "Rainy"):
            weather_rows = [row for row in synthetic_inventory_rows() if row["weather"] == weather]
            for acquisition_date in sorted({row["acquisition_date"] for row in weather_rows})[:3]:
                source.extend(
                    {
                        **row,
                        "subset_id": "synthetic",
                        "split": "unassigned",
                        "selection_seed": "20260902",
                        "date_bucket": "1",
                    }
                    for row in [
                        row
                        for row in weather_rows
                        if row["acquisition_date"] == acquisition_date
                    ][:10]
                )

        first = split_rows(source)
        second = split_rows(source)

        self.assertEqual(first, second)
        validate_splits(first, source)
        self.assertEqual({row["split"] for row in first}, {"smoke", "calibration", "evaluation"})


class DownloadTests(unittest.TestCase):
    def test_safe_destination_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                safe_destination(Path(temporary_directory), Path("../outside.jpg"))

    def test_validate_raw_root_rejects_non_raw_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "data/raw/subset"
            self.assertEqual(validate_raw_root(raw_root, root / "data/raw"), raw_root.resolve())
            with self.assertRaises(ValueError):
                validate_raw_root(root / "outside", root / "data/raw")

    def test_downloader_verifies_images_and_annotation_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            image_path = source / "PKLot/UFPR04/Sunny/2020-01-01/frame.jpg"
            image_path.parent.mkdir(parents=True)
            image_content = b"test image bytes"
            image_path.write_bytes(image_content)
            annotation_json = {"images": [{"file_name": "PKLot/UFPR04/Sunny/2020-01-01/frame.jpg"}]}
            annotation_path = source / "annotations/ufpr04_spots.json"
            annotation_path.parent.mkdir(parents=True)
            annotation_path.write_text(json.dumps(annotation_json), encoding="utf-8")
            archive_path = source / "annotations/ufpr04_spots.tar.xz"
            with tarfile.open(archive_path, "w:xz") as archive:
                archive.add(annotation_path, arcname="ufpr04_spots.json")

            base_url = "https://example.test"
            image_url = f"{base_url}/PKLot/UFPR04/Sunny/2020-01-01/frame.jpg"
            archive_url = f"{base_url}/annotations/ufpr04_spots.tar.xz"
            payloads = {image_url: image_content, archive_url: archive_path.read_bytes()}
            rows = [{
                "image_relpath": "Sunny/2020-01-01/frame.jpg",
                "annotation_relpath": "Sunny/2020-01-01/frame.xml",
                "image_sha256": hashlib.sha256(image_content).hexdigest(),
            }]
            raw_root = root / "data/raw/PKLot/UFPR04"
            with patch(
                "scripts.download_ufpr04_subset.urlopen",
                side_effect=lambda url, timeout: io.BytesIO(payloads[url]),
            ):
                receipt = download_subset(
                    rows,
                    raw_root,
                    f"{base_url}/PKLot/UFPR04",
                    archive_url,
                    hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(receipt["images"]["downloaded"], 1)
                self.assertTrue((raw_root / rows[0]["image_relpath"]).is_file())
                self.assertTrue((raw_root / "_annotations/ufpr04_spots.json").is_file())
                self.assertEqual(
                    download_subset(
                        rows,
                        raw_root,
                        f"{base_url}/PKLot/UFPR04",
                        archive_url,
                        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    )["images"]["verified_existing"],
                    1,
                )


class ParkingSpaceAdapterTests(unittest.TestCase):
    def test_reads_coco_spaces_and_assigns_vehicle_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "ufpr04_spots.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "occupied"}],
                        "images": [
                            {
                                "id": 1,
                                "file_name": "PKLot/UFPR04/Sunny/2020-01-01/frame.jpg",
                                "width": 100,
                                "height": 100,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 10,
                                "image_id": 1,
                                "category_id": 1,
                                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                            },
                            {
                                "id": 20,
                                "image_id": 1,
                                "category_id": 0,
                                "segmentation": [[20, 0, 30, 0, 30, 10, 20, 10]],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            frame = load_frames(annotation_path)["Sunny/2020-01-01/frame.jpg"]

            decisions = assign_occupancy(
                frame.spaces,
                [Detection(1, 1, 9, 9, 0.9, "car"), Detection(20, 0, 25, 5, 0.8, "person")],
                "center",
            )

            self.assertEqual(
                [space.ground_truth for space in frame.spaces], ["occupied", "available"]
            )
            self.assertEqual([decision.status for decision in decisions], ["occupied", "available"])
            self.assertEqual(decisions[0].detector_confidence, 0.9)

    def test_intersection_rule_uses_space_area(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "ufpr04_spots.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "occupied"}],
                        "images": [
                            {
                                "id": 1,
                                "file_name": "PKLot/UFPR04/Sunny/frame.jpg",
                                "width": 100,
                                "height": 100,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 10,
                                "image_id": 1,
                                "category_id": 0,
                                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            space = load_frames(annotation_path)["Sunny/frame.jpg"].spaces
            decisions = assign_occupancy(
                space,
                [Detection(0, 0, 5, 5, 0.7, "truck")],
                "intersection_over_space",
                minimum_overlap_ratio=0.25,
            )
            self.assertEqual(decisions[0].status, "occupied")
            self.assertEqual(decisions[0].overlap_ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
