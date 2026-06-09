"""
train_segmentation.py
---------------------
Trains a YOLO segmentation model on the BDD100K 2-class drivable-area dataset.

Classes
-------
    0 → area/alternative   (secondary / alternate drivable surface)
    1 → area/drivable      (main drivable road surface)

All Colab magic commands removed; runs on any Linux/macOS/Windows machine
with CUDA and the Ultralytics library installed.

Usage:
    python train/train_segmentation.py train \
        --data  /data/BDD100k_yolo_seg_dataset/dataset.yaml \
        --model yolo26l-seg.pt \
        --imgsz 960 \
        --epochs 100 \
        --batch  16 \
        --device 0

    python train/train_segmentation.py eval \
        --weights /data/runs/segment/best.pt \
        --data    /data/BDD100k_yolo_seg_dataset/dataset.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import fiftyone as fo


# ---------------------------------------------------------------------------
# FiftyOne dataset loader (for visual QA only — not required for training)
# ---------------------------------------------------------------------------

def load_fiftyone_dataset(
    images_root: Path,
    labels_root: Path,
    dataset_name: str = "BDD100K_YOLO_SEG_VIS",
    classes: list[str] | None = None,
) -> fo.Dataset:
    """
    Load the YOLO-format segmentation dataset into FiftyOne for visual inspection.

    Parameters
    ----------
    images_root   : Directory containing .jpg / .png images.
    labels_root   : Directory containing YOLO segmentation .txt files.
    dataset_name  : FiftyOne dataset name (recreated if already exists).
    classes       : Class name list ordered by class index.

    Returns
    -------
    fo.Dataset
    """
    classes = classes or ["area/alternative", "area/drivable"]

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    samples: list[fo.Sample] = []

    for img_path in sorted(images_root.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_file = labels_root / f"{img_path.stem}.txt"
        sample = fo.Sample(filepath=str(img_path))
        sample["ground_truth"] = _read_yolo_seg_label(label_file, classes)
        samples.append(sample)

    dataset.add_samples(samples)
    return dataset


def _read_yolo_seg_label(
    label_file: Path,
    classes: list[str],
) -> fo.Polylines:
    polylines: list[fo.Polyline] = []

    if not label_file.exists():
        return fo.Polylines(polylines=[])

    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        if len(coords) % 2 != 0:
            continue

        points = [[x, y] for x, y in zip(coords[0::2], coords[1::2])]
        label  = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"

        polylines.append(fo.Polyline(
            label=label, points=[points], closed=True, filled=True,
        ))

    return fo.Polylines(polylines=polylines)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_yaml: str,
    model_name: str = "yolo26l-seg.pt",
    imgsz: int = 960,
    epochs: int = 100,
    batch: int = 16,
    device: int | str = 0,
    patience: int = 10,
    amp: bool = True,
    run_name: str = "yolo26l_seg_bdd100k",
) -> None:
    """Train a YOLO segmentation model on BDD100K drivable-area data."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.info()

    model.train(
        data     = data_yaml,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        device   = device,
        patience = patience,
        name     = run_name,
        task     = "segment",
        amp      = amp,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    weights: str,
    data_yaml: str,
    save: bool = True,
) -> None:
    """Run YOLO segmentation validation and print mAP metrics."""
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data_yaml, save=save)

    print("\n── Segmentation Evaluation Results ─────────────────")
    print(f"  mAP50-95 (box)  : {metrics.box.map:.4f}")
    print(f"  mAP50    (box)  : {metrics.box.map50:.4f}")
    print(f"  mAP75    (box)  : {metrics.box.map75:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train / evaluate a YOLO segmentation model on BDD100K drivable area."
    )

    subparsers = p.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────
    t = subparsers.add_parser("train", help="Train a YOLO segmentation model.")
    t.add_argument("--data",    required=True, help="Path to dataset.yaml.")
    t.add_argument("--model",   default="yolo26l-seg.pt")
    t.add_argument("--imgsz",   default=960,  type=int)
    t.add_argument("--epochs",  default=100,  type=int)
    t.add_argument("--batch",   default=16,   type=int)
    t.add_argument("--device",  default="0")
    t.add_argument("--patience",default=10,   type=int)
    t.add_argument("--no_amp",  action="store_true",
                   help="Disable Automatic Mixed Precision (use if NaN/Inf warnings appear).")
    t.add_argument("--run_name",default="yolo26l_seg_bdd100k")

    # ── eval ─────────────────────────────────────────────────────────────
    e = subparsers.add_parser("eval", help="Evaluate a trained segmentation model.")
    e.add_argument("--weights", required=True)
    e.add_argument("--data",    required=True)

    # ── visualize ────────────────────────────────────────────────────────
    v = subparsers.add_parser("visualize",
                              help="Launch FiftyOne to visually inspect the dataset.")
    v.add_argument("--images_root", required=True, type=Path)
    v.add_argument("--labels_root", required=True, type=Path)
    v.add_argument("--dataset_name", default="BDD100K_YOLO_SEG_VIS")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "train":
        train(
            data_yaml  = args.data,
            model_name = args.model,
            imgsz      = args.imgsz,
            epochs     = args.epochs,
            batch      = args.batch,
            device     = args.device,
            patience   = args.patience,
            amp        = not args.no_amp,
            run_name   = args.run_name,
        )

    elif args.command == "eval":
        evaluate(weights=args.weights, data_yaml=args.data)

    elif args.command == "visualize":
        dataset = load_fiftyone_dataset(
            images_root  = args.images_root,
            labels_root  = args.labels_root,
            dataset_name = args.dataset_name,
        )
        session = fo.launch_app(dataset)
        session.wait()
