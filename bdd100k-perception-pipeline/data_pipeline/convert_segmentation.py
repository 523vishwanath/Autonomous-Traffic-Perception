"""
convert_segmentation.py
-----------------------
Converts BDD100K poly2d segmentation annotations (JSON) to
YOLO-segmentation .txt labels.

Only drivable-area classes are kept:
    0 → area/alternative   (secondary drivable regions)
    1 → area/drivable      (main drivable road surface)

Handles all known BDD100K poly2d formats:
  • dict  with "vertices" / "closed" keys
  • list  of [x, y, "L"/"C"] triples
  • nested variants of the above

Usage:
    python data_pipeline/convert_segmentation.py \
        --dataset_root /data/BDD100k_sampled \
        --output_root  /data/BDD100k_yolo_seg_dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEEP_CLASSES: frozenset[str] = frozenset({"area/alternative", "area/drivable"})

FIXED_CLASS_MAP: dict[str, int] = {
    "area/alternative": 0,
    "area/drivable":    1,
}

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MIN_POINTS: int = 3


# ---------------------------------------------------------------------------
# JSON / data helpers
# ---------------------------------------------------------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_bdd_objects(data: dict) -> list:
    frames = data.get("frames")
    if not frames:
        return []
    return frames[0].get("objects", [])


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_vertices(
    vertices: list[list[float]], width: int, height: int
) -> list[list[float]]:
    return [[_clamp01(x / width), _clamp01(y / height)] for x, y in vertices]


# ---------------------------------------------------------------------------
# poly2d parser — handles all BDD100K format variants
# ---------------------------------------------------------------------------

def _parse_poly2d(poly2d_data: Any) -> list[dict]:
    """
    Parse a BDD100K poly2d field and return a list of
        {"vertices": [[x, y], ...], "closed": bool}
    dicts, regardless of the source format variant.
    """
    parsed: list[dict] = []

    if poly2d_data is None:
        return parsed

    if isinstance(poly2d_data, dict):
        poly2d_data = [poly2d_data]

    if not isinstance(poly2d_data, list) or len(poly2d_data) == 0:
        return parsed

    first = poly2d_data[0]

    # Direct point-list: [[x, y, "L"], ...] → wrap in outer list
    if (
        isinstance(first, (list, tuple))
        and len(first) >= 2
        and _is_number(first[0])
        and _is_number(first[1])
    ):
        poly2d_data = [poly2d_data]

    for poly in poly2d_data:
        vertices: list[list[float]] = []
        closed = True

        if isinstance(poly, dict):
            closed = bool(poly.get("closed", True))
            for pt in poly.get("vertices", []):
                if (
                    isinstance(pt, (list, tuple))
                    and len(pt) >= 2
                    and _is_number(pt[0])
                    and _is_number(pt[1])
                ):
                    vertices.append([float(pt[0]), float(pt[1])])

        elif isinstance(poly, list):
            for pt in poly:
                if (
                    isinstance(pt, (list, tuple))
                    and len(pt) >= 2
                    and _is_number(pt[0])
                    and _is_number(pt[1])
                ):
                    vertices.append([float(pt[0]), float(pt[1])])

        if vertices:
            parsed.append({"vertices": vertices, "closed": closed})

    return parsed


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_split(
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    class_map: dict[str, int] = FIXED_CLASS_MAP,
    keep_classes: frozenset[str] = KEEP_CLASSES,
    min_points: int = MIN_POINTS,
) -> dict[str, int]:
    """
    Convert one dataset split and return a stats dict.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()

    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    print(f"[INFO] {images_dir.name}: {len(image_files)} images")

    for img_path in image_files:
        json_path = labels_dir / f"{img_path.stem}.json"

        if not json_path.exists():
            stats["missing_json"] += 1
            continue

        try:
            image = cv2.imread(str(img_path))
            if image is None:
                stats["unreadable_image"] += 1
                continue

            h, w = image.shape[:2]
            data = _load_json(json_path)
            objects = _get_bdd_objects(data)

            yolo_lines: list[str] = []

            for obj in objects:
                category = obj.get("category")
                poly2d_raw = obj.get("poly2d")

                if category is None or poly2d_raw is None:
                    continue

                if category not in keep_classes:
                    stats["skipped_other"] += 1
                    continue

                cls_id = class_map[category]

                for poly in _parse_poly2d(poly2d_raw):
                    verts = poly["vertices"]

                    if len(verts) < min_points:
                        stats["too_few_points"] += 1
                        continue

                    norm = _normalize_vertices(verts, w, h)
                    coords = [f"{c:.6f}" for xy in norm for c in xy]

                    if len(coords) < 6:
                        continue

                    yolo_lines.append(f"{cls_id} " + " ".join(coords))
                    stats["written_instances"] += 1

            # Copy image
            shutil.copy2(str(img_path), out_images_dir / img_path.name)
            stats["copied_images"] += 1

            # Write label file (empty labels are valid for background images)
            out_txt = out_labels_dir / f"{img_path.stem}.txt"
            if yolo_lines:
                out_txt.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
                stats["label_files_with_content"] += 1
            else:
                out_txt.write_text("", encoding="utf-8")
                stats["empty_label_files"] += 1

        except Exception as exc:
            print(f"[WARN] Failed {img_path.name}: {exc}")
            stats["errors"] += 1

    return dict(stats)


def convert_all_splits(
    dataset_root: Path,
    output_root: Path,
    splits: list[str] | None = None,
) -> None:
    """Convert every split of the BDD100K drivable-area dataset."""
    splits = splits or ["train", "val", "test"]

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    output_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n[INFO] Split : {split}")

        stats = convert_split(
            images_dir     = images_root / split,
            labels_dir     = labels_root / split,
            out_images_dir = output_root / "images" / split,
            out_labels_dir = output_root / "labels" / split,
        )

        for k, v in sorted(stats.items()):
            print(f"       {k:<35s} = {v}")

    # Write dataset.yaml
    _write_dataset_yaml(output_root)
    print(f"\n[INFO] Segmentation dataset written to : {output_root}")


def _write_dataset_yaml(output_root: Path) -> None:
    names = {v: k for k, v in FIXED_CLASS_MAP.items()}
    lines = [
        f"path: {output_root}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        f"nc: {len(names)}",
        "",
        "names:",
    ]
    for i in sorted(names):
        lines.append(f"  {i}: {names[i]}")

    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] dataset.yaml written to : {yaml_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert BDD100K poly2d drivable-area annotations "
            "to YOLO-segmentation .txt labels."
        )
    )
    p.add_argument("--dataset_root", required=True, type=Path,
                   help="Root folder of the sampled BDD100K dataset "
                        "(must contain images/ and labels/ sub-folders).")
    p.add_argument("--output_root",  required=True, type=Path,
                   help="Destination folder for the YOLO segmentation dataset.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Splits to process (default: train val test).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_all_splits(
        dataset_root = args.dataset_root,
        output_root  = args.output_root,
        splits       = args.splits,
    )
