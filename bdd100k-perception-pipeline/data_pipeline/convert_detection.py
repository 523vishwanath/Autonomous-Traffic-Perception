"""
convert_detection.py
--------------------
Converts BDD100K per-image JSON annotation files into YOLO-format detection
labels (.txt).  Only the nine traffic-relevant object classes are kept.

BDD100K box format : {"x1", "y1", "x2", "y2"}   (pixel coordinates)
YOLO format        : class_id  x_c  y_c  w  h    (normalised 0-1)

Usage:
    python data_pipeline/convert_detection.py \
        --image_root /data/BDD100k_sampled_15k/images \
        --label_root /data/BDD100k_sampled_15k/labels \
        --output_root /data/bdd100k_yoloLabels_15k/labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Class map  (BDD100K category → YOLO class index)
# ---------------------------------------------------------------------------

CLASS_MAP: dict[str, int] = {
    "car":           0,
    "person":        1,
    "rider":         2,
    "truck":         3,
    "bus":           4,
    "motor":         5,
    "bike":          6,
    "traffic light": 7,
    "traffic sign":  8,
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _box_to_yolo(x1: float, y1: float, x2: float, y2: float,
                 img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert pixel-space box corners to normalised YOLO format."""
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw  = (x2 - x1) / img_w
    bh  = (y2 - y1) / img_h
    return xc, yc, bw, bh


def convert_split(
    image_dir: Path,
    label_dir: Path,
    output_dir: Path,
    class_map: dict[str, int] = CLASS_MAP,
    skip_empty: bool = False,
) -> tuple[int, int, int]:
    """
    Convert all JSON labels in *label_dir* that have a matching image in
    *image_dir* and write YOLO .txt files to *output_dir*.

    Returns
    -------
    (converted, empty, missing)
        converted : files where at least one box was written
        empty     : files that existed but had no recognised objects
        missing   : JSON files with no corresponding image (skipped)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = empty = missing = 0

    for json_path in sorted(label_dir.glob("*.json")):
        img_path = image_dir / (json_path.stem + ".jpg")

        if not img_path.exists():
            missing += 1
            continue

        # Image dimensions (needed for normalisation)
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # Parse BDD100K per-image JSON
        with open(json_path) as f:
            data = json.load(f)

        lines: list[str] = []

        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if "box2d" not in obj:
                    continue

                category = obj.get("category", "").lower()
                if category not in class_map:
                    continue

                box = obj["box2d"]
                xc, yc, bw, bh = _box_to_yolo(
                    box["x1"], box["y1"], box["x2"], box["y2"],
                    img_w, img_h,
                )

                cls_id = class_map[category]
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        out_path = output_dir / (json_path.stem + ".txt")

        if lines:
            out_path.write_text("\n".join(lines))
            converted += 1
        else:
            if not skip_empty:
                out_path.write_text("")   # empty label file — valid for YOLO
            empty += 1

    return converted, empty, missing


def convert_all_splits(
    image_root: Path,
    label_root: Path,
    output_root: Path,
    splits: list[str] | None = None,
) -> None:
    """Run conversion for every split (train / val / test)."""
    splits = splits or ["train", "val", "test"]

    for split in splits:
        print(f"\n[INFO] Processing split: {split}")

        conv, emp, miss = convert_split(
            image_dir  = image_root / split,
            label_dir  = label_root / split,
            output_dir = output_root / split,
        )

        print(f"       converted  = {conv}")
        print(f"       empty      = {emp}")
        print(f"       no-image   = {miss}")

    print("\n[INFO] Detection label conversion complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert BDD100K JSON detection labels to YOLO .txt format."
    )
    p.add_argument("--image_root",  required=True, type=Path,
                   help="Root folder containing BDD100K images (split sub-folders).")
    p.add_argument("--label_root",  required=True, type=Path,
                   help="Root folder containing BDD100K JSON label files.")
    p.add_argument("--output_root", required=True, type=Path,
                   help="Output folder for YOLO-format .txt label files.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Splits to process (default: train val test).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_all_splits(
        image_root  = args.image_root,
        label_root  = args.label_root,
        output_root = args.output_root,
        splits      = args.splits,
    )
