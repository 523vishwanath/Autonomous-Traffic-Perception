"""
run_pipeline.py
---------------
End-to-end BDD100K TensorRT perception pipeline.

Combines three real-time perception modules on a driving video:

    1.  Object detection     — TensorRT-optimised YOLO (9 traffic classes)
    2.  Drivable-area segmentation — TensorRT-optimised YOLO-seg (2 classes)
    3.  Monocular depth visualisation — Depth Anything V2 Small (every N frames)

Outputs
-------
    • Annotated output video  (.mp4, 1080p, browser-playable via FFmpeg re-encode)
    • Latency benchmark CSV   (per-module timing across all frames)

Performance (NVIDIA T4 / A100, 1920×1080, imgsz=960):
    ~14–15 FPS  |  det ≈16 ms  |  seg ≈16 ms  |  depth ≈6 ms  |  overlay ≈13 ms

Usage:
    python pipeline/run_pipeline.py \
        --det_engine  /weights/BDD100k_yolov8l_1024_100epochs.engine \
        --seg_engine  /weights/BDD100k_segmentation_yolo26l_2classes_best.engine \
        --input       /videos/driving_clip.mp4 \
        --output_dir  /outputs \
        --imgsz       960 \
        --det_conf    0.35 \
        --seg_conf    0.15

TensorRT export (run once per GPU architecture):
    python pipeline/run_pipeline.py export \
        --det_weights /weights/best_detect.pt \
        --seg_weights /weights/best_seg.pt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Colour palette (one per detection class)
# ---------------------------------------------------------------------------

_DET_COLORS: list[tuple[int, int, int]] = [
    (0,   255,   0),   # car           → Green
    (0,   255, 255),   # person        → Cyan
    (255,   0, 255),   # rider         → Magenta
    (255, 255,   0),   # truck         → Yellow
    (0,   128, 255),   # bus           → Sky Blue
    (255, 128,   0),   # motor         → Orange
    (128, 255,   0),   # bike          → Lime
    (255,   0, 128),   # traffic light → Pink
    (128, 128, 255),   # traffic sign  → Lavender
    (255, 255, 255),   # fallback      → White
]


# ---------------------------------------------------------------------------
# Helper: tensor → numpy
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    det_result,
    use_class_filter: bool = False,
    keep_classes: list[int] | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes + confidence labels on *frame* in-place and return it.
    """
    if det_result is None or det_result.boxes is None:
        return frame

    boxes   = _to_numpy(det_result.boxes.xyxy)
    confs   = _to_numpy(det_result.boxes.conf)
    classes = _to_numpy(det_result.boxes.cls).astype(int)
    names   = det_result.names

    for box, conf, cls in zip(boxes, confs, classes):
        if use_class_filter and keep_classes and cls not in keep_classes:
            continue

        x1, y1, x2, y2 = map(int, box)
        color  = _DET_COLORS[cls % len(_DET_COLORS)]
        label  = f"{names.get(cls, cls)} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = max(y1 - 8, 22)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def apply_seg_overlay(
    frame: np.ndarray,
    seg_result,
    line_width: int = 2,
) -> np.ndarray:
    """
    Use YOLO built-in plot() to overlay segmentation masks.
    Much faster and more accurate than manual polygon rendering.
    """
    if seg_result is None or seg_result.masks is None:
        return frame.copy()

    return seg_result.plot(
        boxes=False, masks=True, probs=False,
        line_width=line_width, font_size=None,
        pil=False, img=frame.copy(),
    )


def make_depth_inset(
    frame: np.ndarray,
    depth_processor,
    depth_model,
    device: str = "cuda",
    input_width: int = 320,
    inset_size: tuple[int, int] = (320, 180),
) -> np.ndarray:
    """
    Compute a Tesla-style grayscale depth map inset from *frame*.

    The frame is first down-sampled to *input_width* to reduce depth model
    inference cost (~4-6 ms at 320 px vs ~60 ms at full 1920 px).
    """
    from PIL import Image

    h, w = frame.shape[:2]
    small_h = int(h * input_width / float(w))

    small_bgr = cv2.resize(frame, (input_width, small_h), interpolation=cv2.INTER_AREA)
    pil_img   = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))

    inputs = depth_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        depth = depth_model(**inputs).predicted_depth

    depth_np = depth.squeeze().detach().float().cpu().numpy()
    depth_8u = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Brighter = closer (Tesla-style)
    depth_gray = cv2.GaussianBlur(255 - depth_8u, (5, 5), 0)
    iw, ih = inset_size
    depth_gray = cv2.resize(depth_gray, (iw, ih), interpolation=cv2.INTER_AREA)

    inset = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

    # Dark header bar
    cv2.rectangle(inset, (0, 0), (iw, 28), (20, 20, 20), -1)
    cv2.putText(inset, "Relative Depth", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(inset, (0, 0), (iw - 1, ih - 1), (200, 200, 200), 1)

    return inset


def paste_inset(
    frame: np.ndarray,
    inset: np.ndarray | None,
    margin: int = 20,
) -> np.ndarray:
    """Paste *inset* in the top-right corner of *frame*."""
    if inset is None:
        return frame

    h, w   = frame.shape[:2]
    ih, iw = inset.shape[:2]
    x1, y1 = w - iw - margin, margin

    if x1 >= 0 and y1 >= 0:
        frame[y1 : y1 + ih, x1 : x1 + iw] = inset

    return frame


def draw_latency_panel(
    frame: np.ndarray,
    instant_fps: float,
    avg_fps: float,
    det_ms: float,
    seg_ms: float,
    depth_ms: float,
    overlay_ms: float,
    total_ms: float,
) -> np.ndarray:
    """Overlay a compact dark latency panel in the top-left corner."""
    lines = [
        f"FPS {instant_fps:.1f}  |  AVG {avg_fps:.1f}",
        f"Det {det_ms:.1f} ms  |  Seg {seg_ms:.1f} ms",
        f"Depth {depth_ms:.1f} ms  |  Overlay {overlay_ms:.1f} ms",
        f"Total {total_ms:.1f} ms",
    ]

    panel_w, panel_h = 480, 120
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (18, 32 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# TensorRT export
# ---------------------------------------------------------------------------

def export_engines(
    det_weights: str,
    seg_weights: str,
    device: int = 0,
) -> None:
    """
    Export PyTorch .pt models to TensorRT .engine files (FP16).

    Run this ONCE on the target GPU; .engine files are architecture-specific.
    """
    from ultralytics import YOLO

    print("[INFO] Exporting detection model …")
    YOLO(det_weights).export(format="engine", half=True, device=device)

    print("[INFO] Exporting segmentation model …")
    YOLO(seg_weights).export(format="engine", half=True, device=device)

    print("[INFO] Export complete.")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_pipeline(
    det_engine: str,
    seg_engine: str,
    input_video: str,
    output_dir: str,
    imgsz: int          = 960,
    det_conf: float     = 0.35,
    seg_conf: float     = 0.15,
    iou: float          = 0.50,
    use_depth: bool     = True,
    depth_every: int    = 5,
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    depth_input_w: int  = 320,
    inset_w: int        = 320,
    inset_h: int        = 180,
    inset_margin: int   = 20,
    use_class_filter: bool = False,
    keep_classes: list[int] | None = None,
    output_scale: float = 1.0,
    draw_latency: bool  = True,
) -> dict:
    """
    Run the full perception pipeline on *input_video*.

    Returns the benchmark dict that is also written to a CSV.
    """
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Validate engine paths ────────────────────────────────────────────
    for path, name in [(det_engine, "Detection"), (seg_engine, "Segmentation")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} engine not found: {path}")

    # ── Load TensorRT engines ────────────────────────────────────────────
    print("[INFO] Loading TensorRT engines …")
    model_det = YOLO(det_engine)
    model_seg = YOLO(seg_engine, task="segment")
    print(f"       Detection classes  : {model_det.names}")
    print(f"       Segmentation classes : {model_seg.names}")

    # ── Load depth model ─────────────────────────────────────────────────
    depth_processor = depth_model = None
    depth_device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_depth:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        print(f"[INFO] Loading depth model: {depth_model_name}")
        depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
        depth_model     = (
            AutoModelForDepthEstimation
            .from_pretrained(depth_model_name)
            .to(depth_device)
            .eval()
        )

        if depth_device == "cuda":
            torch.backends.cudnn.benchmark = True

        print(f"       Depth model on: {depth_device}")
    else:
        print("[INFO] Depth disabled.")

    # ── Open video ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    fps_src     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w, out_h= int(W * output_scale), int(H * output_scale)

    stem = Path(input_video).stem
    raw_output     = os.path.join(output_dir, f"{stem}_pipeline_raw.mp4")
    browser_output = os.path.join(output_dir, f"{stem}_pipeline.mp4")
    benchmark_csv  = os.path.join(results_dir, f"{stem}_benchmark.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output, fourcc, fps_src, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {raw_output}")

    print(f"[INFO] Input  : {W}×{H} @ {fps_src:.1f} FPS  ({total_frames} frames)")
    print(f"[INFO] Output : {out_w}×{out_h}")
    print("[INFO] Running pipeline …\n")

    # ── Timing accumulators ──────────────────────────────────────────────
    det_times = seg_times = depth_times = overlay_times = total_times = fps_list = []
    det_times, seg_times, depth_times, overlay_times, total_times, fps_list = (
        [], [], [], [], [], []
    )

    frame_count    = 0
    last_depth_inset = None
    run_start      = time.time()
    prev_wall      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t_total = time.perf_counter()

        # A. Detection ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        det_result = model_det.predict(
            source=frame, conf=det_conf, iou=iou, imgsz=imgsz,
            verbose=False, device=0, half=True,
            classes=keep_classes if use_class_filter else None,
        )[0]
        det_ms = (time.perf_counter() - t0) * 1e3

        # B. Segmentation ─────────────────────────────────────────────────
        t1 = time.perf_counter()
        seg_result = model_seg.predict(
            source=frame, conf=seg_conf, iou=iou, imgsz=imgsz,
            verbose=False, device=0, half=True,
        )[0]
        seg_ms = (time.perf_counter() - t1) * 1e3

        # C. Depth (every N frames) ────────────────────────────────────────
        t2 = time.perf_counter()
        if use_depth:
            if (
                frame_count == 1
                or last_depth_inset is None
                or frame_count % depth_every == 0
            ):
                last_depth_inset = make_depth_inset(
                    frame, depth_processor, depth_model,
                    device=depth_device, input_width=depth_input_w,
                    inset_size=(inset_w, inset_h),
                )
        depth_ms = (time.perf_counter() - t2) * 1e3

        # D. Overlay ──────────────────────────────────────────────────────
        t3 = time.perf_counter()
        vis = apply_seg_overlay(frame, seg_result)
        vis = draw_detections(vis, det_result, use_class_filter, keep_classes)
        vis = paste_inset(vis, last_depth_inset, margin=inset_margin)
        overlay_ms = (time.perf_counter() - t3) * 1e3

        # E. Latency panel ────────────────────────────────────────────────
        total_ms   = (time.perf_counter() - t_total) * 1e3
        now        = time.time()
        inst_fps   = 1.0 / max(now - prev_wall, 1e-6)
        avg_fps    = frame_count / max(now - run_start, 1e-6)
        prev_wall  = now

        if draw_latency:
            vis = draw_latency_panel(
                vis, inst_fps, avg_fps, det_ms, seg_ms, depth_ms, overlay_ms, total_ms,
            )

        # Resize if needed
        if output_scale != 1.0:
            vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(vis)

        det_times.append(det_ms)
        seg_times.append(seg_ms)
        depth_times.append(depth_ms)
        overlay_times.append(overlay_ms)
        total_times.append(total_ms)
        fps_list.append(inst_fps)

        if frame_count % 30 == 0:
            print(
                f"  Frame {frame_count:5d}/{total_frames} | "
                f"FPS {inst_fps:.1f} | AVG {avg_fps:.1f} | "
                f"Det {det_ms:.0f}ms | Seg {seg_ms:.0f}ms | "
                f"Depth {depth_ms:.0f}ms | Total {total_ms:.0f}ms"
            )

    cap.release()
    writer.release()

    # ── Build benchmark ──────────────────────────────────────────────────
    benchmark = {
        "backend":            "TensorRT + YOLO built-in seg + Depth Anything V2",
        "input_resolution":   f"{W}x{H}",
        "output_resolution":  f"{out_w}x{out_h}",
        "imgsz":              imgsz,
        "depth_every_n":      depth_every if use_depth else 0,
        "avg_det_ms":         round(float(np.mean(det_times)),    2),
        "avg_seg_ms":         round(float(np.mean(seg_times)),    2),
        "avg_depth_ms":       round(float(np.mean(depth_times)),  2),
        "avg_overlay_ms":     round(float(np.mean(overlay_times)),2),
        "avg_total_ms":       round(float(np.mean(total_times)),  2),
        "avg_fps":            round(float(np.mean(fps_list)),     2),
        "median_total_ms":    round(float(np.median(total_times)),2),
        "min_total_ms":       round(float(np.min(total_times)),   2),
        "max_total_ms":       round(float(np.max(total_times)),   2),
        "total_frames":       frame_count,
    }

    print("\n── Benchmark ───────────────────────────────────────────")
    for k, v in benchmark.items():
        print(f"  {k:<26s}: {v}")

    pd.DataFrame([benchmark]).to_csv(benchmark_csv, index=False)
    print(f"\n[INFO] Benchmark CSV : {benchmark_csv}")

    # ── FFmpeg re-encode for browser compatibility ────────────────────────
    ffmpeg_cmd = (
        f'ffmpeg -y -i "{raw_output}" '
        f'-vcodec libx264 -pix_fmt yuv420p -preset fast -crf 23 '
        f'"{browser_output}" -loglevel error'
    )
    print("[INFO] Re-encoding for browser playback …")
    ret = os.system(ffmpeg_cmd)

    if ret == 0:
        print(f"[INFO] Final video : {browser_output}")
    else:
        print(f"[WARN] FFmpeg re-encode failed (exit {ret}). "
              f"Raw output at: {raw_output}")

    return benchmark


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BDD100K TensorRT Perception Pipeline — detection + seg + depth."
    )

    subparsers = p.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────
    r = subparsers.add_parser("run", help="Run the full inference pipeline.")
    r.add_argument("--det_engine",  required=True,
                   help="Path to detection TensorRT .engine file.")
    r.add_argument("--seg_engine",  required=True,
                   help="Path to segmentation TensorRT .engine file.")
    r.add_argument("--input",       required=True,
                   help="Path to input video file.")
    r.add_argument("--output_dir",  default="outputs",
                   help="Output directory (default: outputs/).")
    r.add_argument("--imgsz",       default=960,   type=int)
    r.add_argument("--det_conf",    default=0.35,  type=float)
    r.add_argument("--seg_conf",    default=0.15,  type=float)
    r.add_argument("--iou",         default=0.50,  type=float)
    r.add_argument("--no_depth",    action="store_true",
                   help="Disable depth visualisation.")
    r.add_argument("--depth_every", default=5,     type=int,
                   help="Run depth every N frames (default: 5).")
    r.add_argument("--depth_model", default="depth-anything/Depth-Anything-V2-Small-hf")
    r.add_argument("--output_scale",default=1.0,   type=float,
                   help="Scale output resolution (e.g. 0.5 for half size).")
    r.add_argument("--no_latency",  action="store_true",
                   help="Hide latency panel overlay.")

    # ── export ───────────────────────────────────────────────────────────
    e = subparsers.add_parser("export", help="Export .pt models to TensorRT .engine.")
    e.add_argument("--det_weights", required=True)
    e.add_argument("--seg_weights", required=True)
    e.add_argument("--device",      default=0, type=int)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "run":
        run_pipeline(
            det_engine       = args.det_engine,
            seg_engine       = args.seg_engine,
            input_video      = args.input,
            output_dir       = args.output_dir,
            imgsz            = args.imgsz,
            det_conf         = args.det_conf,
            seg_conf         = args.seg_conf,
            iou              = args.iou,
            use_depth        = not args.no_depth,
            depth_every      = args.depth_every,
            depth_model_name = args.depth_model,
            output_scale     = args.output_scale,
            draw_latency     = not args.no_latency,
        )

    elif args.command == "export":
        export_engines(
            det_weights = args.det_weights,
            seg_weights = args.seg_weights,
            device      = args.device,
        )
