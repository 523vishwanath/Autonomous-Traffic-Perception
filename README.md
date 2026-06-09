<div align="center">

# 🚗 BDD100K Autonomous Driving Perception Pipeline

### TensorRT-Accelerated Object Detection · Drivable-Area Segmentation · Monocular Depth Visualisation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-0057a8?logo=ultralytics&logoColor=white)](https://ultralytics.com)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **Demo** — place your output video here once uploaded to the repo or YouTube.
>
> ```html
> <!-- Option A: GitHub-hosted video (upload the .mp4 to assets/) -->
> <video src="assets/demo.mp4" autoplay loop muted playsinline width="100%"></video>
>
> <!-- Option B: YouTube embed -->
> <!-- [![Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/YOUR_VIDEO_ID) -->
> ```

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Real-World Impact & Applications](#-real-world-impact--applications)
4. [Performance Benchmarks](#-performance-benchmarks)
5. [Repository Structure](#-repository-structure)
6. [Installation](#-installation)
7. [Dataset Preparation](#-dataset-preparation)
8. [Training](#-training)
9. [TensorRT Export](#-tensorrt-export)
10. [Running the Pipeline](#-running-the-pipeline)
11. [Results & Metrics](#-results--metrics)
12. [Technical Stack](#-technical-stack)
13. [Future Work](#-future-work)
14. [Citation](#-citation)

---

## 🎯 Overview

This project implements a **production-grade, multi-task autonomous driving perception pipeline** built on the [BDD100K](https://bdd-data.berkeley.edu/) dataset — one of the largest and most diverse real-world driving datasets in existence (100,000 videos across diverse weather, lighting, and city conditions).

The final system processes a 1080p driving video and simultaneously runs:

| Module | Model | Backend | Latency |
|---|---|---|---|
| Object Detection | YOLO26l (9 classes) | TensorRT FP16 | ~16 ms |
| Drivable-Area Segmentation | YOLO26l-seg (2 classes) | TensorRT FP16 | ~16 ms |
| Monocular Depth Visualisation | Depth Anything V2 Small | PyTorch (CUDA) | ~6 ms |
| Full Pipeline (1080p) | All 3 combined | GPU | **~51 ms (~14–15 FPS)** |

### Detection Classes (9)

`car` · `person` · `rider` · `truck` · `bus` · `motorcycle` · `bike` · `traffic light` · `traffic sign`

### Segmentation Classes (2)

`area/drivable` (main road surface) · `area/alternative` (secondary drivable area)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION PIPELINE                        │
│                                                                      │
│  BDD100K Raw Dataset                                                 │
│  (100k images, JSON annotations)                                     │
│           │                                                          │
│           ▼                                                          │
│  dataset_split.py          ──── Sample 15k images (70/10/20 split)  │
│           │                                                          │
│      ┌────┴────┐                                                     │
│      ▼         ▼                                                     │
│  convert_        convert_                                            │
│  detection.py    segmentation.py                                     │
│  (JSON → YOLO    (poly2d → YOLO                                      │
│   box labels)     seg labels)                                        │
│      │               │                                               │
│      ▼               ▼                                               │
│  build_yolo_structure.py   ──── Organise into Ultralytics YOLO dir  │
└──────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                             │
│                                                                      │
│   train_detection.py         train_segmentation.py                  │
│   YOLO26l · imgsz=960        YOLO26l-seg · imgsz=960                │
│   epochs=100 · batch=32      epochs=100  · batch=16                 │
│           │                          │                               │
│           ▼                          ▼                               │
│   best_detect.pt             best_seg.pt                            │
└──────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     TENSORRT DEPLOYMENT                              │
│                                                                      │
│   .pt  ──export──▶  .engine (FP16, GPU-specific)                    │
│                                                                      │
│   Detection Engine  +  Segmentation Engine  +  Depth Anything V2    │
│                                                                      │
│                  run_pipeline.py                                     │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │  Input Frame (1920×1080)                                   │    │
│   │        │                                                   │    │
│   │   ┌────┴──────┬─────────────┬──────────────────┐          │    │
│   │   ▼           ▼             ▼ (every 5 frames)  │          │    │
│   │  TRT Det    TRT Seg      Depth Anything V2       │          │    │
│   │  16 ms      16 ms        6 ms                    │          │    │
│   │   │           │             │                    │          │    │
│   │   └────────────────────────┘                     │          │    │
│   │                 │                                            │    │
│   │          Overlay Renderer  ──── 13 ms             │          │    │
│   │      (boxes + seg + depth inset + FPS panel)     │          │    │
│   │                 │                                            │    │
│   │          Output Frame + CSV Benchmark             │          │    │
│   └────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🌍 Real-World Impact & Applications

This pipeline is not a research demo — it mirrors the actual perception stack used in production autonomous driving systems and advanced driver-assistance systems (ADAS). Here is where this directly applies:

### 🚘 Autonomous Vehicle Perception (L2–L4)
Companies like **Waymo, Tesla, Cruise, Mobileye, and Zoox** run exactly this stack in production: simultaneous multi-task inference (detection + segmentation + depth) on GPU or custom NPU hardware. This project demonstrates the core building blocks of how those systems work, including the dataset engineering, multi-task model training, and deployment-side optimisation via TensorRT.

### 🛡️ Advanced Driver Assistance Systems (ADAS)
Modern vehicles (BMW, Mercedes, Toyota, Hyundai) use onboard ADAS chips (Mobileye EyeQ, NVIDIA Orin) to run real-time perception nearly identical to this pipeline. Drivable-area segmentation directly enables lane-keeping, adaptive cruise control, and emergency braking. The 9-class detection covers every object class relevant to collision avoidance.

### 🚦 Smart Traffic Infrastructure
City traffic management systems increasingly use roadside cameras with perception models to:
- Count vehicle types and traffic density in real time
- Detect near-miss events and pedestrian violations
- Optimize traffic signal timing dynamically

The detection and segmentation models here are directly applicable to such deployments.

### 🚌 Fleet Safety & Dashcam AI
Commercial vehicle fleets (trucking, delivery, ride-share) use onboard perception AI for:
- Driver safety scoring
- Near-miss and incident detection
- Insurance telematics

This pipeline runs on standard NVIDIA GPUs at 14–15 FPS on 1080p — well within the requirements for dashcam AI processing.

### 🤖 Robotics & Industrial Automation
The same perception stack generalises to warehouse robots, autonomous forklifts, and last-mile delivery robots that need to navigate in shared human environments. Drivable-area segmentation maps directly to "navigable floor surface" for ground robots.

### 🏥 Emergency Response & Search Operations
Perception systems on emergency vehicles and drones use detection models to identify people, vehicles, and hazards in complex urban environments. The person and rider classes in this model are directly relevant to such applications.

### 📐 Why This Matters for AI/ML Engineers

This project demonstrates a critical real-world insight that is often missing from research papers:

> **A production computer-vision system is not just about model accuracy. The bottleneck is almost never the model itself — it is preprocessing, postprocessing, visualization, memory bandwidth, and I/O throughput.**

This pipeline went from **3.3 FPS → 14 FPS** without any change to the model architecture, purely through pipeline-level engineering: removing redundant rendering, batching depth computation, and using YOLO's built-in plot() instead of custom polygon drawing. That kind of systems thinking is what distinguishes production engineers from research engineers.

---

## 📊 Performance Benchmarks

All benchmarks measured on 1920×1080 video, imgsz=960, NVIDIA T4 GPU (Google Colab).

### Final Optimised Pipeline

| Metric | Value |
|---|---|
| Backend | TensorRT FP16 + Depth Anything V2 |
| Input Resolution | 1920×1080 |
| Output Resolution | 1920×1080 |
| Image Size (inference) | 960 |
| Detection Latency | **16.17 ms** |
| Segmentation Latency | **15.68 ms** |
| Depth Latency (avg, every 5 frames) | **6.28 ms** |
| Overlay Latency | **13.05 ms** |
| **Total Latency (avg)** | **51.18 ms** |
| **End-to-End FPS** | **~14–15 FPS** |
| Median Total Latency | 44.68 ms |
| Frames Processed | 1,798 |

### Optimisation Journey

| Version | Output Size | Depth | Seg Renderer | FPS |
|---|---|---|---|---|
| v1 — Naive | 3840×1080 | Full screen (every frame) | Custom polygon | ~3.3 |
| v2 — Side-by-side removed | 1920×1080 | Full screen | Custom polygon | ~6.0 |
| v3 — Depth down-sampled | 1920×1080 | Inset (320px input) | Custom polygon | ~8.5 |
| v4 — YOLO built-in seg | 1920×1080 | Inset | **YOLO plot()** | ~11.0 |
| **v5 — Depth every 5 frames** | **1920×1080** | **Inset (every 5)** | YOLO plot() | **~14–15** |

**4× improvement** purely through pipeline engineering — no model changes.

---

## 📁 Repository Structure

```
bdd100k-perception-pipeline/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_pipeline/              # BDD100K → YOLO dataset preparation
│   ├── dataset_split.py        # Sample & split BDD100K (15k images, 70/10/20)
│   ├── convert_detection.py    # BDD100K JSON boxes → YOLO .txt labels
│   ├── convert_segmentation.py # BDD100K poly2d masks → YOLO seg .txt labels
│   ├── build_yolo_structure.py # Reorganise into Ultralytics YOLO folder layout
│   └── check_dataset.py        # Validate image-label pairs & YOLO value ranges
│
├── train/                      # Model training scripts
│   ├── train_detection.py      # Train YOLO26l detection (9 traffic classes)
│   └── train_segmentation.py   # Train YOLO26l-seg segmentation (2 classes)
│
├── pipeline/                   # Inference & deployment
│   └── run_pipeline.py         # TensorRT video pipeline + benchmark CSV
│
└── assets/                     # Demo media for this README
    └── demo.mp4                # ← upload your output video here
```

---

## ⚙️ Installation

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU with CUDA ≥ 11.8
- cuDNN ≥ 8.x
- For TensorRT inference: TensorRT ≥ 8.6

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/bdd100k-perception-pipeline.git
cd bdd100k-perception-pipeline

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install TensorRT (required for .engine inference)
pip install tensorrt
```

---

## 📥 Dataset Preparation

### 1. Download BDD100K

Register and download from the [official BDD100K website](https://bdd-data.berkeley.edu/).

You need:
- `bdd100k_images_100k.zip` — 100k driving images
- `bdd100k_labels_release.zip` — Detection JSON annotations
- `bdd100k_drivable_labels_trainval.zip` — Drivable area segmentation labels

### 2. Sample & Split

```bash
python data_pipeline/dataset_split.py \
    --image_root /data/BDD100k/100k_images \
    --label_root /data/BDD100k/100k \
    --output_root /data/BDD100k_sampled_15k \
    --sample_size 15000 \
    --train 10500 --val 1500 --test 3000 \
    --seed 42
```

### 3. Convert Detection Labels (JSON → YOLO)

BDD100K stores bounding boxes as `{x1, y1, x2, y2}` pixel coordinates.
This script converts them to normalised YOLO format `class_id x_c y_c w h`.

```bash
python data_pipeline/convert_detection.py \
    --image_root  /data/BDD100k_sampled_15k/images \
    --label_root  /data/BDD100k_sampled_15k/labels \
    --output_root /data/BDD100k_sampled_15k/yolo_labels
```

### 4. Convert Segmentation Labels (poly2d → YOLO-seg)

BDD100K drivable-area annotations are stored as polygon lists.
This script normalises the polygon vertices and writes YOLO segmentation labels.

```bash
python data_pipeline/convert_segmentation.py \
    --dataset_root /data/BDD100k_sampled \
    --output_root  /data/BDD100k_yolo_seg_dataset
```

### 5. Build Final YOLO Directory Layout

```bash
python data_pipeline/build_yolo_structure.py \
    --src_root    /data/BDD100k_sampled_15k \
    --label_root  /data/BDD100k_sampled_15k/yolo_labels \
    --output_root /data/bdd100k_yoloLabels_15k
```

### 6. Validate Dataset Integrity

```bash
python data_pipeline/check_dataset.py \
    --image_dir /data/bdd100k_yoloLabels_15k/train/images \
    --label_dir /data/bdd100k_yoloLabels_15k/train/labels
```

Expected output:
```
── Detection label report ──────────────────────────
  total               : 10500
  matched             : 10497
  empty               : 823
  bad_values          : 0
  missing_label       : 3
```

### Dataset Summary

| Split | Images | Purpose |
|---|---|---|
| Train | 10,500 | Model training |
| Val | 1,500 | Training-time evaluation |
| Test | 3,000 | Final benchmark |
| **Total** | **15,000** | |

---

## 🏋️ Training

### Detection Model

```bash
python train/train_detection.py train \
    --data    /data/bdd100k_yoloLabels_15k/dataset.yaml \
    --model   yolo26l.pt \
    --imgsz   960 \
    --epochs  100 \
    --batch   16 \
    --device  0

# Evaluate
python train/train_detection.py eval \
    --weights runs/detect/train/weights/best.pt \
    --data    /data/bdd100k_yoloLabels_15k/dataset.yaml

# Visualise training annotations
python train/train_detection.py visualize \
    --image_dir /data/bdd100k_yoloLabels_15k/train/images \
    --label_dir /data/bdd100k_yoloLabels_15k/train/labels \
    --num 40
```

### Training Configuration (Detection)

| Parameter | Value | Notes |
|---|---|---|
| Base model | `yolo26l.pt` | Large-scale, fine-tuned from COCO |
| Image size | 960 | Good balance of accuracy and speed |
| Batch size | 16–32 | Adjust to VRAM |
| Epochs | 100 | With `patience=15` early stopping |
| AMP | ✅ | Mixed precision — ~2× faster |
| HSV augmentation | h=0.014, s=0.7, v=0.5 | Rain/night adaptation |

### Segmentation Model

```bash
python train/train_segmentation.py train \
    --data     /data/BDD100k_yolo_seg_dataset/dataset.yaml \
    --model    yolo26l-seg.pt \
    --imgsz    960 \
    --epochs   100 \
    --batch    16 \
    --device   0 \
    --run_name yolo26l_seg_bdd100k

# Evaluate
python train/train_segmentation.py eval \
    --weights runs/segment/yolo26l_seg_bdd100k/weights/best.pt \
    --data    /data/BDD100k_yolo_seg_dataset/dataset.yaml

# Visually inspect dataset (FiftyOne)
python train/train_segmentation.py visualize \
    --images_root /data/BDD100k_yolo_seg_dataset/images/train \
    --labels_root /data/BDD100k_yolo_seg_dataset/labels/train
```

### Training Configuration (Segmentation)

| Parameter | Value |
|---|---|
| Base model | `yolo26l-seg.pt` |
| Task | `segment` |
| Classes | 2 (alternative, drivable) |
| Image size | 960 |
| Epochs | 100 |
| AMP | ⚠️ Disable if NaN/Inf warnings (`--no_amp`) |

---

## ⚡ TensorRT Export

TensorRT engines are **GPU-architecture-specific**. Export once on your target GPU (T4, A100, Orin, etc.).

```bash
python pipeline/run_pipeline.py export \
    --det_weights runs/detect/train/weights/best.pt \
    --seg_weights runs/segment/yolo26l_seg_bdd100k/weights/best.pt \
    --device 0
```

This produces:
- `best_detect.engine` — Detection TensorRT FP16 engine
- `best_seg.engine` — Segmentation TensorRT FP16 engine

> **Note:** `.engine` files are excluded from git (see `.gitignore`). Upload them to Google Drive or an S3 bucket and reference them by path.

---

## 🎬 Running the Pipeline

```bash
python pipeline/run_pipeline.py run \
    --det_engine  /weights/BDD100k_detect.engine \
    --seg_engine  /weights/BDD100k_seg.engine \
    --input       /videos/driving_clip.mp4 \
    --output_dir  outputs/ \
    --imgsz       960 \
    --det_conf    0.35 \
    --seg_conf    0.15
```

### All Options

| Flag | Default | Description |
|---|---|---|
| `--det_engine` | — | Path to detection `.engine` |
| `--seg_engine` | — | Path to segmentation `.engine` |
| `--input` | — | Input video path |
| `--output_dir` | `outputs/` | Output directory |
| `--imgsz` | `960` | Inference image size |
| `--det_conf` | `0.35` | Detection confidence threshold |
| `--seg_conf` | `0.15` | Segmentation confidence threshold |
| `--iou` | `0.50` | NMS IoU threshold |
| `--no_depth` | — | Disable depth visualisation |
| `--depth_every` | `5` | Run depth every N frames |
| `--output_scale` | `1.0` | Resize output (e.g. `0.5` for half) |
| `--no_latency` | — | Hide latency overlay panel |

### Outputs

```
outputs/
├── <stem>_pipeline_raw.mp4       # Raw pipeline output (mp4v codec)
├── <stem>_pipeline.mp4           # Browser-playable H.264 re-encode
└── results/
    └── <stem>_benchmark.csv      # Per-frame latency statistics
```

---

## 📈 Results & Metrics

### Object Detection (mAP on BDD100K test set)

| Class | AP50 |
|---|---|
| car | — |
| person | — |
| rider | — |
| truck | — |
| bus | — |
| motorcycle | — |
| bike | — |
| traffic light | — |
| traffic sign | — |
| **mAP50 (all)** | **— (fill in after eval)** |
| **mAP50-95** | **—** |

> Fill in your evaluation results from `python train/train_detection.py eval ...`

### Drivable-Area Segmentation

| Metric | Value |
|---|---|
| mAP50 (box) | ~95+ |
| mAP50-95 (box) | ~70+ |
| Classes | 2 |

### Output Frame

The final video frame contains:
- **Detection boxes** — coloured per class with confidence scores
- **Segmentation overlay** — semi-transparent drivable-area masks (YOLO built-in rendering)
- **Depth inset** — top-right grayscale relative depth map (Depth Anything V2, Tesla-style)
- **Latency panel** — top-left dark panel showing per-module and total FPS/latency

---

## 🔧 Technical Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.1+ |
| **Detection / Segmentation** | Ultralytics YOLO (YOLO26l, YOLO26l-seg) |
| **Depth Estimation** | Depth Anything V2 Small (HuggingFace Transformers) |
| **Deployment / Acceleration** | NVIDIA TensorRT FP16 |
| **Video I/O** | OpenCV |
| **Dataset Visualisation** | FiftyOne |
| **Data Format** | BDD100K JSON → YOLO .txt |
| **Hardware** | NVIDIA GPU (T4 / A100 / Orin) |
| **Re-encoding** | FFmpeg (H.264 / yuv420p) |
| **Benchmarking** | Pandas CSV + per-frame `time.perf_counter()` |

---

## 🔮 Future Work

- [ ] **Object tracking** — Integrate BoT-SORT or ByteTrack for persistent IDs across frames
- [ ] **Near-car distance estimation** — Aggregate depth values inside detection boxes to rank proximity of vehicles
- [ ] **Lane detection** — Add lane-line segmentation (BDD100K provides lane annotations)
- [ ] **Jetson Orin deployment** — Rebuild `.engine` for NVIDIA Jetson Orin NX for edge deployment
- [ ] **Risk scoring** — Assign real-time risk levels (low/medium/high) to detected objects based on size and proximity
- [ ] **Per-frame depth caching** — Evaluate interpolation-based approaches to smooth depth between updates
- [ ] **Model distillation** — Distil YOLO26l into a smaller model for constrained hardware
- [ ] **30 FPS target** — Profile remaining overlay overhead; implement CUDA-accelerated blending

---

## 📝 Citation

If you use this work, please cite the BDD100K dataset and Depth Anything V2:

```bibtex
@InProceedings{bdd100k,
    author    = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and
                 Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title     = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {CVPR},
    year      = {2020}
}

@article{depth_anything_v2,
    title   = {Depth Anything V2},
    author  = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and
               Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
    journal = {arXiv:2406.09414},
    year    = {2024}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

The BDD100K dataset is subject to its own [terms of use](https://bdd-data.berkeley.edu/portal.html#terms).
YOLO model weights used for fine-tuning are subject to [Ultralytics licensing](https://ultralytics.com/license).

---

<div align="center">

**Built with ❤️ for autonomous driving research and real-world deployment**

</div>
