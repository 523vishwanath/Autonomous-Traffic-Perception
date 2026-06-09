"""
train_detection.py
------------------
Trains a YOLO detection model on the BDD100K 9-class traffic dataset.

Supports YOLO11 / YOLO26 model variants and is fully configurable via
command-line arguments.  All Colab-specific magic commands have been removed;
the script runs on any machine with an NVIDIA GPU and the Ultralytics library.

Usage:
    python train/train_detection.py \
        --data     /data/bdd100k_yoloLabels_15k/dataset.yaml \
        --model    yolo26l.pt \
        --imgsz    960 \
        --epochs   100 \
        --batch    16 \
        --device   0

Visualise training results:
    python train/train_detection.py --visualize \
        --run_dir runs/detect/train
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Class map & colour palette
# ---------------------------------------------------------------------------

CLASS_MAP: dict[str, int] = {
    "car":           0,
    "person":        1,
    "rider":         2,
    "truck":         3,
    "bus":           4,
    "motor":         5,
    "bike":          6,
    "traffic_light": 7,
    "traffic_sign":  8,
}

ID2COLOR: dict[int, tuple[int, int, int]] = {
    0: (45,  123, 200),   # car           → Steel Blue
    1: (255,   0,   0),   # person        → Red
    2: (142, 204,  88),   # rider         → Lime Green
    3: (189,  40, 215),   # truck         → Purple
    4: (67,  210, 156),   # bus           → Turquoise
    5: (105, 105, 105),   # motorcycle    → Dim Gray
    6: (230, 190, 255),   # bike          → Lavender
    7: (0,   255, 255),   # traffic light → Cyan
    8: (250, 128, 114),   # traffic sign  → Salmon
}


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def visualize_with_bboxes(
    image_folder: Path,
    label_folder: Path,
    num_images: int = 40,
    color_map: dict[int, tuple[int, int, int]] = ID2COLOR,
) -> None:
    """Display a random grid of annotated training images."""
    id2name = {v: k for k, v in CLASS_MAP.items()}

    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".png"))
    ]
    selected = random.sample(image_files, min(num_images, len(image_files)))

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
    axes = axes.ravel()

    for i, fname in enumerate(selected):
        img = cv2.imread(str(image_folder / fname))
        label_path = label_folder / (Path(fname).stem + ".txt")

        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id, xc, yc, bw, bh = int(float(parts[0])), *map(float, parts[1:5])

                x1 = int((xc - bw / 2) * w_img)
                y1 = int((yc - bh / 2) * h_img)
                x2 = int((xc + bw / 2) * w_img)
                y2 = int((yc + bh / 2) * h_img)

                r, g, b = color_map.get(cls_id, (255, 255, 255))
                bgr = (b, g, r)

                cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    img,
                    id2name.get(cls_id, str(cls_id)),
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, bgr, 1, cv2.LINE_AA,
                )

        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("BDD100K — Sample Training Annotations", fontsize=16)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_yaml: str,
    model_name: str = "yolo26l.pt",
    imgsz: int = 960,
    epochs: int = 100,
    batch: int = 16,
    device: int | str = 0,
    patience: int = 15,
    amp: bool = True,
    cache: bool = False,
) -> None:
    """Train a YOLO detection model on BDD100K."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.info()

    model.train(
        data    = data_yaml,
        imgsz   = imgsz,
        batch   = batch,
        epochs  = epochs,
        amp     = amp,
        device  = device,
        cache   = cache,
        patience = patience,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    weights: str,
    data_yaml: str,
    save: bool = True,
) -> None:
    """Run YOLO validation and print mAP metrics."""
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data_yaml, save=save)

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP75    : {metrics.box.map75:.4f}")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def run_inference(
    image_paths: list[str],
    model_path: str,
    output_dir: str = "inference_results",
    device: str = "cpu",
    conf: float = 0.25,
    iou: float = 0.45,
) -> None:
    """Run detection inference on a list of images and save annotated outputs."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        results = model.predict(
            source  = img_path,
            conf    = conf,
            iou     = iou,
            device  = device,
            save    = False,
            verbose = False,
        )
        plotted = results[0].plot()
        save_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, plotted)

    print(f"[INFO] Inference complete. Results saved to : {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train / evaluate a YOLO detection model on BDD100K."
    )

    subparsers = p.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────
    t = subparsers.add_parser("train", help="Train a YOLO detection model.")
    t.add_argument("--data",    required=True, help="Path to dataset.yaml.")
    t.add_argument("--model",   default="yolo26l.pt",
                   help="YOLO model to fine-tune (default: yolo26l.pt).")
    t.add_argument("--imgsz",   default=960,  type=int,
                   help="Training image size (default: 960).")
    t.add_argument("--epochs",  default=100,  type=int)
    t.add_argument("--batch",   default=16,   type=int)
    t.add_argument("--device",  default="0")
    t.add_argument("--patience",default=15,   type=int)
    t.add_argument("--no_amp",  action="store_true",
                   help="Disable Automatic Mixed Precision.")

    # ── eval ─────────────────────────────────────────────────────────────
    e = subparsers.add_parser("eval", help="Evaluate a trained YOLO model.")
    e.add_argument("--weights", required=True, help="Path to .pt weights file.")
    e.add_argument("--data",    required=True, help="Path to dataset.yaml.")

    # ── visualize ────────────────────────────────────────────────────────
    v = subparsers.add_parser("visualize",
                              help="Visualise training images with ground-truth boxes.")
    v.add_argument("--image_dir", required=True, type=Path)
    v.add_argument("--label_dir", required=True, type=Path)
    v.add_argument("--num",       default=40, type=int,
                   help="Number of images to display (default: 40).")

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
        )

    elif args.command == "eval":
        evaluate(weights=args.weights, data_yaml=args.data)

    elif args.command == "visualize":
        visualize_with_bboxes(
            image_folder = args.image_dir,
            label_folder = args.label_dir,
            num_images   = args.num,
        )
