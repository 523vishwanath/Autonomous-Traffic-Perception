"""
build_yolo_structure.py
-----------------------
Reorganises a sampled BDD100K dataset from the intermediate layout produced by
``dataset_split.py`` + ``convert_detection.py`` into the flat YOLO layout
expected by Ultralytics:

    <output_root>/
        train/
            images/  <── .jpg files
            labels/  <── YOLO .txt files
        val/
            images/
            labels/
        test/
            images/
            labels/

Usage:
    python data_pipeline/build_yolo_structure.py \
        --src_root    /data/BDD100k_sampled_15k \
        --label_root  /data/BDD100k_sampled_15k/yolo_labels \
        --output_root /data/bdd100k_yoloLabels_15k
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_yolo_structure(
    src_root: Path,
    label_root: Path,
    output_root: Path,
    splits: list[str] | None = None,
) -> None:
    """
    Copy images and YOLO labels from *src_root* / *label_root* into a clean
    Ultralytics YOLO directory layout under *output_root*.

    Parameters
    ----------
    src_root    : Directory containing ``images/<split>/`` sub-folders.
    label_root  : Directory containing YOLO .txt label sub-folders
                  (one per split, names matching image stems).
    output_root : Destination root for the final YOLO dataset.
    splits      : List of split names to process.
    """
    splits = splits or ["train", "val", "test"]

    # Create destination directories
    for split in splits:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] YOLO structure created under : {output_root}")

    total_images = 0
    total_missing_labels = 0

    for split in splits:
        img_src  = src_root    / "images" / split
        lbl_src  = label_root  / split
        img_dst  = output_root / split / "images"
        lbl_dst  = output_root / split / "labels"

        if not img_src.exists():
            print(f"[WARN] Image source not found, skipping : {img_src}")
            continue

        image_files = sorted(
            f for f in img_src.iterdir() if f.suffix.lower() in {".jpg", ".png"}
        )

        copied_imgs  = 0
        missing_lbls = 0

        for img_path in image_files:
            label_name = img_path.stem + ".txt"
            lbl_path   = lbl_src / label_name

            shutil.copy2(img_path, img_dst / img_path.name)
            copied_imgs += 1

            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / label_name)
            else:
                # Write empty label so YOLO does not error on missing file
                (lbl_dst / label_name).write_text("")
                missing_lbls += 1

        print(
            f"[INFO] {split:5s} : {copied_imgs} images, "
            f"{missing_lbls} labels auto-created as empty"
        )

        total_images        += copied_imgs
        total_missing_labels += missing_lbls

    print(f"\n[INFO] Done.  Total images: {total_images}  |  "
          f"Auto-empty labels: {total_missing_labels}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reorganise BDD100K data into Ultralytics YOLO folder structure."
    )
    p.add_argument("--src_root",    required=True, type=Path,
                   help="Root of the source dataset (contains images/<split>/).")
    p.add_argument("--label_root",  required=True, type=Path,
                   help="Root of YOLO .txt label files (contains <split>/).")
    p.add_argument("--output_root", required=True, type=Path,
                   help="Destination root for the YOLO-layout dataset.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Splits to process (default: train val test).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_yolo_structure(
        src_root    = args.src_root,
        label_root  = args.label_root,
        output_root = args.output_root,
        splits      = args.splits,
    )
