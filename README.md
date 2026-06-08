# BDD100K Autonomous Driving Perception Pipeline

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Optimized-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-00CFDD?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**TensorRT-optimized multi-task perception at 14+ FPS on 1080p driving video.**  
Object detection · Drivable-area segmentation · Monocular depth visualization · Real-time latency benchmarking.

</div>

---

## Demo

> **To embed your output video:** upload `bdd_tensorrt_yolo_builtin_seg_depth_browser.mp4` directly to this GitHub README via the edit interface (drag and drop), or convert 10 seconds to a GIF using the command below and place it at `assets/demo.gif`.

```bash
# Convert your output video to an auto-playing GIF for the README
ffmpeg -i outputs/bdd_tensorrt_yolo_builtin_seg_depth_browser.mp4 \
       -t 10 \
       -vf "fps=12,scale=960:-1:flags=lanczos" \
       -loop 0 \
       assets/demo.gif
```

<!-- Replace this block with your embedded video or GIF once exported -->
```
assets/demo.gif   ← drop your demo GIF here
```
<!-- ![Inference Demo](assets/demo.gif) -->

*1920×1080 input · YOLO object detection · drivable-area segmentation · Depth Anything V2 inset · real-time FPS/latency overlay*

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Performance](#performance)
- [Dataset Preparation](#dataset-preparation)
- [Models](#models)
- [Training](#training)
- [TensorRT Deployment](#tensorrt-deployment)
- [Optimization Journey](#optimization-journey)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Configuration Reference](#configuration-reference)
- [Technical Stack](#technical-stack)
- [Future Work](#future-work)

---

## Overview

This project builds a **production-style autonomous driving perception pipeline** on top of the [BDD100K](https://bdd-data.berkeley.edu/) dataset — one of the largest and most diverse real-world driving datasets available.

The system takes a dashcam driving video as input and produces an annotated output video with:

| Layer | What it does |
|---|---|
| **Object Detection** | Localizes cars, pedestrians, cyclists, buses, trucks, traffic lights, and traffic signs with per-class confidence scores |
| **Drivable-Area Segmentation** | Segments the road into *main drivable* and *alternative drivable* regions using polygon-level masks |
| **Depth Visualization** | Overlays a Tesla-style grayscale monocular depth inset (closer = brighter) using Depth Anything V2 |
| **Latency Panel** | Displays per-module inference time and rolling average FPS directly on every frame |

Both the detection and segmentation models are converted to **TensorRT FP16 engines** for optimized GPU inference.  
The full pipeline runs end-to-end at **~14–15 FPS on 1920×1080 video**.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     INPUT VIDEO (1920×1080)                │
└─────────────────────────┬─────────────────────────────────┘
                          │ frame
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
  ┌───────────────┐ ┌───────────────┐ ┌──────────────────┐
  │  TensorRT     │ │  TensorRT     │ │ Depth Anything   │
  │  Detection    │ │  Segmentation │ │ V2 Small (HF)    │
  │  Engine       │ │  Engine       │ │ every N frames   │
  │               │ │               │ │                  │
  │ YOLOv8l-det   │ │ YOLO26l-seg   │ │ 320px input      │
  │ 9 classes     │ │ 2 classes     │ │ 320×180 inset    │
  │ ~16 ms/frame  │ │ ~16 ms/frame  │ │ ~6 ms/frame avg  │
  └───────┬───────┘ └───────┬───────┘ └────────┬─────────┘
          │                 │                   │
          └─────────────────┼───────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Visualization Layer   │
              │                         │
              │  YOLO built-in plot()   │  ← segmentation masks
              │  Manual bbox drawing    │  ← detection boxes
              │  Depth inset paste      │  ← top-right corner
              │  Latency panel          │  ← top-left corner
              │                         │
              │  ~13 ms/frame           │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │  OUTPUT VIDEO (1920×1080)│
              │  FFmpeg → browser MP4   │
              │  Benchmark CSV          │
              └─────────────────────────┘
```

---

## Performance

### Final Benchmark (Optimized Pipeline)

| Metric | Value |
|---|---|
| Input resolution | 1920 × 1080 |
| Output resolution | 1920 × 1080 |
| Inference image size | 960 px |
| Detection latency | **16.17 ms** |
| Segmentation latency | **15.68 ms** |
| Depth latency (avg, every 5 frames) | **6.28 ms** |
| Overlay latency | **13.05 ms** |
| **End-to-end latency** | **51.18 ms** |
| **Average FPS** | **14.27** |
| Median frame latency | 44.68 ms |
| Total frames processed | 1,798 |

### Optimization Journey — Before vs After

| Stage | Bottleneck | Avg FPS | Notes |
|---|---|---|---|
| **v1 — Initial** | 3840×1080 side-by-side output, full-screen depth, heavy custom mask renderer | **3.34 FPS** | Custom contour/polygon drawing + alpha blending caused 168 ms overlay cost |
| **v2 — Optimized** | Eliminated all above bottlenecks | **14.27 FPS** | **4.3× improvement** with same TensorRT models |

The TensorRT models were already fast at v1 (~16 ms each). The bottleneck was entirely in the **visualization layer**, not in inference — a key real-world deployment insight.

---

## Dataset Preparation

The BDD100K dataset contains 100,000 driving images across diverse conditions. This project filters, converts, and structures a usable 15K-image subset for training.

The full data preparation pipeline lives in `data_prep/` and runs in five stages:

### Stage 1 — Verify Image-Label Pairs
**Script:** `data_prep/bdd100k_files_checking.py`

Scans the raw BDD100K image directory and JSON label directory to count valid image-label pairs before committing to a large copy operation.

### Stage 2 — Sample and Split
**Script:** `data_prep/dataset_split.py`

Randomly samples 15,000 images from the full 100K dataset and splits them deterministically (seed = 42):

| Split | Count | Share |
|---|---|---|
| Train | 10,500 | 70% |
| Val | 1,500 | 10% |
| Test | 3,000 | 20% |

### Stage 3 — BDD100K JSON → YOLO Detection Labels
**Script:** `data_prep/yoloformat_conversion.py`

Reads BDD100K per-frame JSON annotations and converts bounding boxes to normalized YOLO format for 9 detection classes:

```
# BDD100K format                 # YOLO format
{                                 class_id  x_center  y_center  width  height
  "box2d": {                  →   0         0.512345  0.334567  0.123456  0.089012
    "x1": 612, "y1": 287,
    "x2": 854, "y2": 433
  }
}
```

**Detection class map:**

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | car | 5 | motorcycle |
| 1 | person | 6 | bike |
| 2 | rider | 7 | traffic light |
| 3 | truck | 8 | traffic sign |
| 4 | bus | | |

### Stage 4 — Build YOLO Folder Structure
**Script:** `data_prep/yolo_fileStructure.py`

Re-organizes the converted images and labels into the standard YOLO directory layout expected by Ultralytics:

```
bdd100k_yoloLabels_15k/
├── train/
│   ├── images/   (10,500 JPGs)
│   └── labels/   (10,500 TXTs)
├── val/
│   ├── images/   (1,500 JPGs)
│   └── labels/   (1,500 TXTs)
├── test/
│   ├── images/   (3,000 JPGs)
│   └── labels/   (3,000 TXTs)
└── dataset.yaml
```

### Stage 5 — BDD100K Poly2D → YOLO Segmentation Labels
**Script:** `data_prep/extract_segment_poly2d.py`

Extracts `area/drivable` and `area/alternative` polygon annotations from BDD100K segmentation JSONs and converts them to YOLO segmentation format (normalized polygon point sequences):

```
# YOLO segmentation label format
# class_id  x1 y1  x2 y2  x3 y3  ...  (all normalized 0–1)
1  0.231  0.512  0.345  0.489  0.412  0.534  ...
```

The parser handles all four BDD100K polygon2D encoding variants, clamps out-of-bound coordinates, filters degenerate polygons with fewer than 3 points, and skips non-drivable classes.

**Segmentation class map:**

| ID | Class | Description |
|---|---|---|
| 0 | `area/alternative` | Non-primary drivable region (side roads, shoulders) |
| 1 | `area/drivable` | Primary drivable road surface |

### Stage 6 — Verify Segmentation Masks
**Script:** `data_prep/drivable_masks_checking.py`

Cross-validates that every sampled image has a corresponding segmentation mask before training begins.

---

## Models

### Detection Model — `YOLOv8l`

| Parameter | Value |
|---|---|
| Architecture | YOLOv8l (large) |
| Training images | 10,500 |
| Input size | 1024 px (training) / 960 px (inference) |
| Epochs | 100 |
| Batch size | 32 |
| Classes | 9 (car, person, rider, truck, bus, motorcycle, bike, traffic light, traffic sign) |
| Augmentation | HSV jitter (h=0.014, s=0.7, v=0.5) |
| Deployment | TensorRT FP16 engine |
| Inference latency | ~16 ms @ 1080p |

**Color scheme used for visualization:**

```python
id2color = {
    0: (45, 123, 200),   # car          — Steel Blue
    1: (255, 0,   0  ),  # person       — Red
    2: (142, 204, 88 ),  # rider        — Lime Green
    3: (189, 40,  215),  # truck        — Purple
    4: (67,  210, 156),  # bus          — Turquoise
    5: (105, 105, 105),  # motorcycle   — Dim Gray
    6: (230, 190, 255),  # bike         — Lavender
    7: (0,   255, 255),  # traffic light— Cyan
    8: (250, 128, 114),  # traffic sign — Salmon
}
```

### Segmentation Model — `YOLO26l-seg`

| Parameter | Value |
|---|---|
| Architecture | YOLO26l-seg |
| Input size | 960 px |
| Epochs | 100 |
| Batch size | 16 |
| Classes | 2 (area/alternative, area/drivable) |
| Task | Instance segmentation (polygon masks) |
| Reported performance | mAP50 ≈ 95+ · mAP50-95 ≈ 70+ |
| Deployment | TensorRT FP16 engine |
| Inference latency | ~16 ms @ 1080p |

### Depth Model — `Depth Anything V2 Small`

| Parameter | Value |
|---|---|
| Model | `depth-anything/Depth-Anything-V2-Small-hf` |
| Output | Relative monocular depth map (not metric distance) |
| Input (pipeline) | Resized to 320 px wide for speed |
| Inset size | 320 × 180 px (top-right corner) |
| Run frequency | Every 5 frames (cached in between) |
| Avg latency | ~6.28 ms (amortized over 5 frames) |
| Style | Tesla-style inverted grayscale (bright = near, dark = far) |

---

## Training

### Detection Training

```bash
# Train YOLOv8l on BDD100K detection subset
# Requires: bdd100k_yoloLabels_15k/ with dataset.yaml

python training/bdd_detect_train.py
```

The training script:
1. Unzips the prepared dataset
2. Writes `dataset.yaml` with correct paths and class names
3. Optionally visualizes random training images with bounding boxes via FiftyOne
4. Trains `yolo26l.pt` with AMP, patience=15, imgsz=960
5. Saves the best checkpoint to Google Drive

### Segmentation Training

```bash
# Train YOLO26l-seg on BDD100K drivable-area segmentation subset
# Requires: BDD100k_yolo_seg_dataset/ with dataset.yaml

python training/bdd_lane_segmentation.py
```

The training script:
1. Unzips the segmentation dataset
2. Writes `dataset.yaml` for 2-class drivable-area task
3. Optionally visualizes polygon annotations via FiftyOne
4. Trains `yolo26l-seg.pt` with `task="segment"`, AMP disabled (stability), patience=10
5. Saves best checkpoint to Google Drive

---

## TensorRT Deployment

After training, export `.pt` models to TensorRT FP16 engines once:

```python
from ultralytics import YOLO

# Detection
model = YOLO("BDD100k_yolov8l_1024_100epochs.pt")
model.export(format="engine", half=True, device=0)

# Segmentation
model = YOLO("BDD100k_segmentation_yolo26l_2classes_best.pt")
model.export(format="engine", half=True, device=0, task="segment")
```

The inference pipeline (`pipeline/bdd100k_pipeline.py`) **loads engines directly** — no re-export at inference time:

```python
model_detect_trt  = YOLO("BDD100k_yolov8l_1024_100epochs.engine")
model_segment_trt = YOLO("BDD100k_segmentation_yolo26l_2classes_best.engine", task="segment")
```

> **Note:** TensorRT engines are GPU-architecture specific. Rebuild engines if you change GPU hardware.

### Running the Pipeline

```bash
python pipeline/bdd100k_pipeline.py
```

Outputs:
- `outputs/bdd_tensorrt_yolo_builtin_seg_depth.mp4` — raw output
- `outputs/bdd_tensorrt_yolo_builtin_seg_depth_browser.mp4` — browser-playable (FFmpeg H.264)
- `results/benchmark_yolo_builtin_seg_depth.csv` — per-run benchmark

---

## Optimization Journey

This section documents the engineering decisions that drove the **4.3× FPS improvement** from v1 to the final pipeline. Understanding where the bottleneck actually lives is the core lesson.

### What was slow in v1

| Component | v1 Latency | Problem |
|---|---|---|
| Detection (TensorRT) | ~16.8 ms | ✅ Already fast — not the bottleneck |
| Segmentation (TensorRT) | ~16.1 ms | ✅ Already fast — not the bottleneck |
| Depth | ~61.0 ms | Running full-resolution, every frame |
| **Overlay rendering** | **~168.7 ms** | **The real bottleneck** |
| **Total** | **~266.6 ms** | **3.34 FPS** |

The overlay took 168 ms because it was drawing polygon contours, alpha-blending segmentation masks, and manually resizing mask tensors for a **3840×1080** frame.

### Changes made in the final version

| Change | Impact |
|---|---|
| Removed side-by-side 3840×1080 layout → single 1920×1080 frame | Halved all per-pixel operations |
| Replaced custom mask renderer with `seg_result.plot()` | Overlay: 168 ms → 13 ms |
| Moved depth to 320×180 top-right inset | Eliminated full-frame depth resize |
| Run depth every 5 frames, reuse cached inset | Depth: 61 ms → 6.28 ms avg |
| Resized depth input to 320 px wide before model | Faster depth forward pass |
| Removed TensorRT export from inference notebook | Cleaner startup, no accidental re-export |

---

## Project Structure

```
bdd100k-perception-pipeline/
│
├── data_prep/                        # Dataset preparation utilities
│   ├── bdd100k_files_checking.py     # Verify image-label pair counts
│   ├── dataset_split.py              # Sample 15K and split 70/10/20
│   ├── yoloformat_conversion.py      # BDD100K JSON → YOLO detection labels
│   ├── yolo_fileStructure.py         # Build YOLO folder structure
│   ├── drivable_masks_checking.py    # Verify segmentation masks
│   └── extract_segment_poly2d.py     # BDD100K Poly2D → YOLO seg labels
│
├── training/
│   ├── bdd_detect_train.py           # YOLOv8l detection training on BDD100K
│   └── bdd_lane_segmentation.py      # YOLO26l-seg drivable-area training
│
├── pipeline/
│   └── bdd100k_pipeline.py           # TensorRT video inference pipeline
│
├── assets/
│   └── demo.gif                      # Demo GIF for README (generate with FFmpeg)
│
├── outputs/                          # Generated at runtime
│   ├── bdd_tensorrt_yolo_builtin_seg_depth.mp4
│   └── bdd_tensorrt_yolo_builtin_seg_depth_browser.mp4
│
├── results/                          # Generated at runtime
│   └── benchmark_yolo_builtin_seg_depth.csv
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- CUDA-capable NVIDIA GPU (TensorRT requires NVIDIA GPU)
- CUDA 11.8+ and cuDNN installed
- TensorRT 8.x+ installed system-wide

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics lapx transformers accelerate tabulate
pip install pandas==2.2.2 numpy==2.0.0 Pillow==11.3.0
pip install tensorrt  # requires NVIDIA GPU
```

### 1. Prepare the Dataset

Run the data preparation scripts in order:

```bash
# Step 1 — verify pairs
python data_prep/bdd100k_files_checking.py

# Step 2 — sample 15K and split
python data_prep/dataset_split.py

# Step 3 — convert detection labels to YOLO format
python data_prep/yoloformat_conversion.py

# Step 4 — build YOLO folder structure
python data_prep/yolo_fileStructure.py

# Step 5 — extract segmentation polygons
python data_prep/extract_segment_poly2d.py

# Step 6 — verify segmentation masks
python data_prep/drivable_masks_checking.py
```

Update the path constants at the top of each script to point to your local BDD100K download before running.

### 2. Train Detection Model

```bash
python training/bdd_detect_train.py
```

### 3. Train Segmentation Model

```bash
python training/bdd_lane_segmentation.py
```

### 4. Export to TensorRT

```python
from ultralytics import YOLO

YOLO("weights/detection_best.pt").export(format="engine", half=True, device=0)
YOLO("weights/segmentation_best.pt").export(format="engine", half=True, device=0, task="segment")
```

### 5. Run the Inference Pipeline

Edit `DETECTION_ENGINE_PATH`, `SEGMENTATION_ENGINE_PATH`, and `INPUT_VIDEO_PATH` in `pipeline/bdd100k_pipeline.py`, then:

```bash
python pipeline/bdd100k_pipeline.py
```

---

## Configuration Reference

All inference settings are controlled by the constants at the top of `pipeline/bdd100k_pipeline.py`:

```python
# ── Engine paths ─────────────────────────────────────────────────────────────
DETECTION_ENGINE_PATH     = "BDD100k_yolov8l_1024_100epochs.engine"
SEGMENTATION_ENGINE_PATH  = "BDD100k_segmentation_yolo26l_2classes_best.engine"

# ── Input / output ───────────────────────────────────────────────────────────
INPUT_VIDEO_PATH          = "your_driving_video.mp4"

# ── Inference ────────────────────────────────────────────────────────────────
IMG_SIZE                  = 960       # YOLO inference image size
DETECTION_CONF            = 0.35      # detection confidence threshold
SEGMENT_CONF              = 0.15      # segmentation confidence threshold
IOU_THRESH                = 0.50      # NMS IoU threshold

# ── Depth ────────────────────────────────────────────────────────────────────
USE_DEPTH                 = True
DEPTH_EVERY_N_FRAMES      = 5         # run depth every N frames
DEPTH_INPUT_WIDTH         = 320       # resize input before depth model
DEPTH_INSET_WIDTH         = 320       # inset pixel width
DEPTH_INSET_HEIGHT        = 180       # inset pixel height

# ── Visualization ────────────────────────────────────────────────────────────
USE_YOLO_BUILTIN_SEG_PLOT = True      # use YOLO plot() instead of custom renderer
DRAW_DETECTIONS_MANUALLY  = True      # draw colored bounding boxes
DRAW_LATENCY_PANEL        = True      # top-left FPS / ms overlay
OUTPUT_SCALE              = 1.0       # set < 1.0 to downscale output (e.g. 0.5 for 960×540)
```

---

## Technical Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep learning framework | PyTorch 2.x |
| Detection / segmentation | Ultralytics YOLO (YOLOv8l, YOLO26l-seg) |
| Depth estimation | Depth Anything V2 Small (HuggingFace Transformers) |
| Deployment / optimization | NVIDIA TensorRT (FP16, `.engine` format) |
| GPU acceleration | CUDA 11.8+ |
| Video I/O | OpenCV, FFmpeg |
| Dataset visualization | FiftyOne |
| Annotation conversion | NumPy, Pillow, json |
| Benchmarking | Pandas |
| Training environment | Google Colab (NVIDIA A100) |

---

## Limitations

| Limitation | Details |
|---|---|
| **Depth is relative, not metric** | Depth Anything V2 outputs relative depth; no true distance in meters |
| **Engine portability** | TensorRT engines are GPU-architecture specific — must rebuild on a different GPU |
| **FPS headroom** | Pipeline runs at ~14 FPS; real-time 30 FPS would require further optimization (e.g. smaller model, lower resolution, full CUDA stream pipelining) |
| **Segmentation scope** | Only 2 drivable-area classes — no full scene parsing (sidewalk, sky, building, etc.) |
| **No tracking** | Objects are detected per-frame with no temporal association |

---

## Future Work

- [ ] **Near-vehicle distance estimation** — aggregate depth values inside detection bounding boxes to produce relative proximity scores for each detected vehicle
- [ ] **Object tracking** — integrate BoT-SORT or ByteTrack for consistent object IDs across frames
- [ ] **Segmentation interpolation** — run segmentation every 2–3 frames and warp masks between frames using optical flow for higher effective FPS
- [ ] **Lane line detection** — extend segmentation to include lane marking annotations from BDD100K
- [ ] **Edge deployment** — export and benchmark on NVIDIA Jetson Orin or Xavier NX
- [ ] **Risk scoring** — assign proximity-based risk labels (safe / caution / danger) to nearby objects using relative depth
- [ ] **Per-module latency dashboard** — interactive Streamlit or Gradio dashboard for profiling
- [ ] **H100-optimized engines** — rebuild TensorRT engines with INT8 calibration for maximum throughput

---

## Citation

If you use this project or find it useful, please cite:

```bibtex
@misc{bdd100k-perception-pipeline,
  title   = {BDD100K Autonomous Driving Perception Pipeline},
  author  = {Vishwanath Reddy},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```

Dataset:
```bibtex
@inproceedings{bdd100k,
  author    = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  title     = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  booktitle = {CVPR},
  year      = {2020}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The BDD100K dataset is subject to its own [license](https://bdd-data.berkeley.edu/). This repository does not distribute any dataset files.
