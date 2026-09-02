#!/usr/bin/env python3
"""Render annotated contact sheets for manual review of a PKLot subset."""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.inventory_pklot import WEATHERS

THUMBNAIL_SIZE = (320, 180)
GRID_COLUMNS = 5
HEADER_HEIGHT = 28
CAPTION_HEIGHT = 22


def render_thumbnail(image_path: Path, annotation_path: Path, caption: str) -> Image.Image:
    with Image.open(image_path) as source:
        source.load()
        image = source.convert("RGB").resize(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(image)
    scale_x = THUMBNAIL_SIZE[0] / 1280
    scale_y = THUMBNAIL_SIZE[1] / 720
    root = ET.parse(annotation_path).getroot()
    for space in root.findall("space"):
        contour = space.find("contour")
        point_elements = (
            [child for child in contour if child.tag.lower() == "point"]
            if contour is not None
            else []
        )
        points = [
            (int(point.attrib["x"]) * scale_x, int(point.attrib["y"]) * scale_y)
            for point in point_elements
        ]
        label = space.get("occupied")
        if label == "1":
            color = (239, 68, 68)
        elif label == "0":
            color = (34, 197, 94)
        else:
            color = (234, 179, 8)
        if len(points) >= 3:
            draw.line([*points, points[0]], fill=color, width=1)

    tile = Image.new("RGB", (THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1] + CAPTION_HEIGHT), "white")
    tile.paste(image, (0, 0))
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.text((5, THUMBNAIL_SIZE[1] + 4), caption, fill="black", font=ImageFont.load_default())
    return tile


def create_contact_sheet(
    rows: list[dict[str, str]], subset_root: Path, output_path: Path, weather: str
) -> None:
    tile_height = THUMBNAIL_SIZE[1] + CAPTION_HEIGHT
    grid_rows = (len(rows) + GRID_COLUMNS - 1) // GRID_COLUMNS
    sheet = Image.new(
        "RGB",
        (GRID_COLUMNS * THUMBNAIL_SIZE[0], HEADER_HEIGHT + grid_rows * tile_height),
        (240, 240, 240),
    )
    draw = ImageDraw.Draw(sheet)
    title = (
        f"UFPR04 balanced v1: {weather} ({len(rows)} frames) | "
        "red=occupied green=vacant yellow=unlabeled"
    )
    draw.text(
        (8, 8),
        title,
        fill="black",
        font=ImageFont.load_default(),
    )

    for index, row in enumerate(rows):
        caption = (
            f"{row['acquisition_date']} {row['capture_time']} "
            f"occ={row['occupied_count']}/{row['labeled_space_count']}"
        )
        tile = render_thumbnail(
            subset_root / row["image_relpath"],
            subset_root / row["annotation_relpath"],
            caption,
        )
        x = (index % GRID_COLUMNS) * THUMBNAIL_SIZE[0]
        y = HEADER_HEIGHT + (index // GRID_COLUMNS) * tile_height
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "JPEG", quality=90, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pklot_ufpr04_subset_v1.csv"),
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("data/processed/ufpr04_balanced_v1"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qa"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for weather in WEATHERS:
        weather_rows = [row for row in rows if row["weather"] == weather]
        create_contact_sheet(
            weather_rows,
            args.subset_root,
            args.output_dir / f"ufpr04_balanced_v1_{weather.lower()}.jpg",
            weather,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
