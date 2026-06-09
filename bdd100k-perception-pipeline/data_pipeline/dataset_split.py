"""
dataset_split.py
----------------
Randomly samples a fixed number of images from the full BDD100K dataset
and splits them into train / val / test subsets, copying both the images
and their JSON annotation files into a clean output directory.

Usage:
    python data_pipeline/dataset_split.py \
        --image_root /data/BDD100k/100k_images \
        --label_root /data/BDD100k/100k \
        --output_root /data/BDD100k_sampled_15k \
        --sample_size 15000 \
        --train 10500 \
        --val 1500 \
        --test 3000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def split_dataset(
    image_root: Path,
    label_root: Path,
    output_root: Path,
    sample_size: int = 15_000,
    train_size: int = 10_500,
    val_size: int = 1_500,
    test_size: int = 3_000,
    random_seed: int = 42,
) -> None:
    """
    Sample *sample_size* images from *image_root*, split them into train/val/test,
    and copy image + label pairs to *output_root*.

    Parameters
    ----------
    image_root  : Root directory that contains the raw BDD100K images
                  (sub-folders 'train', 'val', etc. are walked recursively).
    label_root  : Root directory that contains per-image JSON annotation files
                  mirroring the structure of *image_root*.
    output_root : Destination directory for the sampled YOLO-ready dataset.
    sample_size : Total number of images to sample.
    train_size  : Number allocated to the training split.
    val_size    : Number allocated to the validation split.
    test_size   : Number allocated to the test split.
    random_seed : Random seed for reproducibility.
    """
    assert train_size + val_size + test_size == sample_size, (
        "train + val + test must equal sample_size"
    )

    all_images = sorted(image_root.rglob("*.jpg"))
    print(f"[INFO] Total images found : {len(all_images)}")

    if len(all_images) < sample_size:
        raise ValueError(
            f"Only {len(all_images)} images available but {sample_size} requested."
        )

    random.seed(random_seed)
    sampled = random.sample(all_images, sample_size)

    splits = {
        "train": sampled[:train_size],
        "val":   sampled[train_size : train_size + val_size],
        "test":  sampled[train_size + val_size :],
    }

    for split, imgs in splits.items():
        print(f"[INFO] {split:5s} : {len(imgs)} images")

    # Create output directories
    for split in splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for split, img_list in splits.items():
        img_out = output_root / "images" / split
        lbl_out = output_root / "labels" / split
        copied = 0
        missing = 0

        for img_path in img_list:
            label_path = label_root / img_path.relative_to(image_root).with_suffix(".json")

            if not label_path.exists():
                print(f"[WARN] Missing label : {label_path.name}")
                missing += 1
                continue

            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)
            copied += 1

        print(f"[INFO] {split:5s} : {copied} pairs copied, {missing} labels missing")

    print(f"\n[INFO] Dataset written to : {output_root}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample and split BDD100K into train/val/test subsets."
    )
    p.add_argument("--image_root",  required=True, type=Path,
                   help="Root folder containing BDD100K images (searched recursively).")
    p.add_argument("--label_root",  required=True, type=Path,
                   help="Root folder containing BDD100K per-image JSON labels.")
    p.add_argument("--output_root", required=True, type=Path,
                   help="Output folder for the sampled dataset.")
    p.add_argument("--sample_size", default=15_000, type=int,
                   help="Total number of images to sample (default: 15000).")
    p.add_argument("--train",  default=10_500, type=int, dest="train_size",
                   help="Training set size (default: 10500).")
    p.add_argument("--val",    default=1_500,  type=int, dest="val_size",
                   help="Validation set size (default: 1500).")
    p.add_argument("--test",   default=3_000,  type=int, dest="test_size",
                   help="Test set size (default: 3000).")
    p.add_argument("--seed",   default=42,     type=int,
                   help="Random seed (default: 42).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    split_dataset(
        image_root  = args.image_root,
        label_root  = args.label_root,
        output_root = args.output_root,
        sample_size = args.sample_size,
        train_size  = args.train_size,
        val_size    = args.val_size,
        test_size   = args.test_size,
        random_seed = args.seed,
    )
