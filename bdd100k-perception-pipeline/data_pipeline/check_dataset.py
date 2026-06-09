"""
check_dataset.py
----------------
Verifies dataset integrity before training.

Checks performed
----------------
1.  Image–label pair matching     : every image has a corresponding label file.
2.  YOLO label validation         : all bounding-box values are in [0, 1].
3.  Empty label count             : images without annotations are reported.
4.  Drivable-area mask alignment  : (optional) checks that BDD100K colour
    segmentation masks exist for sampled images.

Usage:
    # Check detection dataset
    python data_pipeline/check_dataset.py \
        --image_dir /data/bdd100k_yoloLabels_15k/train/images \
        --label_dir /data/bdd100k_yoloLabels_15k/train/labels

    # Also verify colour-mask availability
    python data_pipeline/check_dataset.py \
        --image_dir /data/bdd100k_yoloLabels_15k/train/images \
        --label_dir /data/bdd100k_yoloLabels_15k/train/labels \
        --mask_dir  /data/bdd100k/bdd100k_seg_maps/color_labels/train
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_detection_labels(
    image_dir: Path,
    label_dir: Path,
) -> dict[str, int]:
    """
    Validate YOLO detection label files against their images.

    Returns a summary dict with keys:
        total, matched, empty, bad_values, missing_label
    """
    stats = {
        "total":         0,
        "matched":       0,
        "empty":         0,
        "bad_values":    0,
        "missing_label": 0,
    }

    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for img_path in image_files:
        stats["total"] += 1
        label_path = label_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            stats["missing_label"] += 1
            continue

        stats["matched"] += 1
        lines = label_path.read_text().splitlines()

        if not lines or all(line.strip() == "" for line in lines):
            stats["empty"] += 1
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                values = list(map(float, line.split()))
                # YOLO box: class x_c y_c w h — coords must be 0-1
                if any(v > 1.0 or v < 0.0 for v in values[1:]):
                    stats["bad_values"] += 1
                    break
            except ValueError:
                stats["bad_values"] += 1
                break

    return stats


def check_mask_alignment(
    image_dir: Path,
    mask_dir: Path,
    mask_suffix: str = "_train_color.png",
) -> dict[str, int]:
    """
    Count how many images in *image_dir* have a corresponding colour-mask
    file in *mask_dir* following BDD100K naming convention:
        <stem><mask_suffix>
    """
    stats = {"images": 0, "with_mask": 0, "missing_mask": 0}

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        stats["images"] += 1
        mask_path = mask_dir / (img_path.stem + mask_suffix)

        if mask_path.exists():
            stats["with_mask"] += 1
        else:
            stats["missing_mask"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify BDD100K YOLO dataset integrity."
    )
    p.add_argument("--image_dir", required=True, type=Path,
                   help="Directory containing .jpg images.")
    p.add_argument("--label_dir", required=True, type=Path,
                   help="Directory containing YOLO .txt label files.")
    p.add_argument("--mask_dir",  default=None, type=Path,
                   help="(Optional) BDD100K colour-mask directory to check alignment.")
    p.add_argument("--mask_suffix", default="_train_color.png",
                   help="Suffix appended to image stem to find the mask file.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"[INFO] Checking detection labels in : {args.label_dir}")
    det_stats = check_detection_labels(args.image_dir, args.label_dir)

    print("\n── Detection label report ──────────────────────────")
    for k, v in det_stats.items():
        print(f"  {k:<20s}: {v}")

    if args.mask_dir:
        print(f"\n[INFO] Checking mask alignment in : {args.mask_dir}")
        mask_stats = check_mask_alignment(args.image_dir, args.mask_dir, args.mask_suffix)

        print("\n── Mask alignment report ───────────────────────────")
        for k, v in mask_stats.items():
            print(f"  {k:<20s}: {v}")
