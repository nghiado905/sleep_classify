from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DETECT_MODEL = PROJECT_ROOT / "human-detection" / "models" / "human_detection_2class.pt"
DEFAULT_CLASSIFY_MODEL = PROJECT_ROOT / "runs" / "model_ngu_gat" / "yolov11m_cls_30k" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
TARGET_LABELS = {"normal", "sleep"}
LABEL_COLORS = {
    "normal": (55, 180, 80),
    "sleep": (40, 80, 240),
    "unknown": (180, 180, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer anh/video: detect nguoi, classify normal/sleep, ve bbox + nhan va luu output."
    )
    parser.add_argument("source", type=Path, help="Duong dan anh, thu muc anh/video, hoac video")
    parser.add_argument("--detect-model", type=Path, default=DEFAULT_DETECT_MODEL, help="YOLO detection model .pt")
    parser.add_argument("--cls-model", type=Path, default=DEFAULT_CLASSIFY_MODEL, help="YOLO classification model .pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thu muc ket qua")
    parser.add_argument("--det-imgsz", type=int, default=640, help="Kich thuoc anh cho detector")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="Kich thuoc anh cho classifier")
    parser.add_argument("--conf", type=float, default=0.25, help="Nguong confidence detector")
    parser.add_argument("--iou", type=float, default=0.7, help="Nguong IoU NMS detector")
    parser.add_argument(
        "--device",
        default="0",
        help="Thiet bi inference (mac dinh GPU CUDA 0). Vi du: 0, 1, cpu",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Video: classify moi N frame, frame giua se dung lai ket qua gan nhat",
    )
    return parser.parse_args()


def collect_sources(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        )
    return []


def unique_destination(directory: Path, source: Path, suffix: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    extension = suffix or source.suffix.lower()
    destination = directory / f"{source.stem}{extension}"
    index = 1
    while destination.exists():
        destination = directory / f"{source.stem}_{index:04d}{extension}"
        index += 1
    return destination


def ensure_label_dirs(output_dir: Path) -> None:
    for label in sorted(TARGET_LABELS):
        (output_dir / label).mkdir(parents=True, exist_ok=True)


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(value)) for value in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def classify_crop(
    cls_model: YOLO,
    crop: np.ndarray,
    imgsz: int,
    device: str | None,
) -> tuple[str, float]:
    result = cls_model.predict(
        source=crop,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    if result.probs is None:
        raise RuntimeError("Classification model khong tra ve probs. Hay dung model *-cls.pt.")

    top_index = int(result.probs.top1)
    label = str(result.names[top_index]).strip().lower()
    confidence = float(result.probs.top1conf)
    if label not in TARGET_LABELS:
        label = "unknown"
    return label, confidence


def draw_label(frame: np.ndarray, box: tuple[int, int, int, int], label: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
    text = f"{label} {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    label_y1 = max(0, y1 - text_height - baseline - 6)
    label_y2 = label_y1 + text_height + baseline + 6
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 8, label_y2), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, label_y2 - baseline - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def dominant_label(labels: list[str]) -> str:
    valid = [label for label in labels if label in TARGET_LABELS]
    if not valid:
        return "normal"
    return Counter(valid).most_common(1)[0][0]


def annotate_frame(
    frame: np.ndarray,
    det_model: YOLO,
    cls_model: YOLO,
    args: argparse.Namespace,
    cached_predictions: list[tuple[tuple[int, int, int, int], str, float]] | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[tuple[int, int, int, int], str, float]]]:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    predictions: list[tuple[tuple[int, int, int, int], str, float]] = []

    if cached_predictions is None:
        result = det_model.predict(
            source=frame,
            imgsz=args.det_imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        if result.boxes is not None:
            for raw_box in result.boxes.xyxy.cpu().numpy():
                box = clip_box(raw_box, width, height)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                label, confidence = classify_crop(cls_model, crop, args.cls_imgsz, args.device)
                predictions.append((box, label, confidence))
    else:
        predictions = cached_predictions

    labels: list[str] = []
    for box, label, confidence in predictions:
        draw_label(annotated, box, label, confidence)
        labels.append(label)
    return annotated, labels, predictions


def process_image(
    source: Path,
    det_model: YOLO,
    cls_model: YOLO,
    args: argparse.Namespace,
) -> dict[str, str]:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Khong doc duoc anh: {source}")

    annotated, labels, _ = annotate_frame(frame, det_model, cls_model, args)
    label = dominant_label(labels)
    destination = unique_destination(args.output / label, source)
    cv2.imwrite(str(destination), annotated)

    return {
        "source": str(source),
        "output": str(destination),
        "type": "image",
        "label": label,
        "detections": str(len(labels)),
        "normal": str(labels.count("normal")),
        "sleep": str(labels.count("sleep")),
    }


def process_video(
    source: Path,
    det_model: YOLO,
    cls_model: YOLO,
    args: argparse.Namespace,
) -> dict[str, str]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_dir = args.output / "_tmp"
    temp_destination = unique_destination(temp_dir, source, ".mp4")
    writer = cv2.VideoWriter(
        str(temp_destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Khong tao duoc video output: {temp_destination}")

    all_labels: list[str] = []
    cached_predictions: list[tuple[tuple[int, int, int, int], str, float]] | None = None
    sample_rate = max(1, args.sample_rate)

    progress = tqdm(total=frame_count or None, desc=f"Video {source.name}")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_infer = frame_index % sample_rate == 0 or cached_predictions is None
            annotated, labels, predictions = annotate_frame(
                frame,
                det_model,
                cls_model,
                args,
                None if should_infer else cached_predictions,
            )
            if should_infer:
                cached_predictions = predictions

            all_labels.extend(labels)
            writer.write(annotated)
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()
        writer.release()

    label = dominant_label(all_labels)
    destination = unique_destination(args.output / label, source, ".mp4")
    temp_destination.replace(destination)

    return {
        "source": str(source),
        "output": str(destination),
        "type": "video",
        "label": label,
        "detections": str(len(all_labels)),
        "normal": str(all_labels.count("normal")),
        "sleep": str(all_labels.count("sleep")),
    }


def run(args: argparse.Namespace) -> int:
    args.source = args.source.resolve()
    args.detect_model = args.detect_model.resolve()
    args.cls_model = args.cls_model.resolve()
    args.output = args.output.resolve()

    if not args.detect_model.is_file():
        print(f"[LOI] Khong tim thay detect model: {args.detect_model}", file=sys.stderr)
        return 1
    if not args.cls_model.is_file():
        print(f"[LOI] Khong tim thay classify model: {args.cls_model}", file=sys.stderr)
        return 1

    sources = collect_sources(args.source)
    if not sources:
        print(f"[LOI] Khong tim thay anh/video hop le trong: {args.source}", file=sys.stderr)
        return 1

    ensure_label_dirs(args.output)
    csv_path = args.output / "predictions.csv"

    print("=" * 90)
    print(" SLEEP/NORMAL VIDEO-IMAGE INFERENCE ".center(90, "="))
    print(f"  Source       : {args.source}")
    print(f"  Detect model : {args.detect_model}")
    print(f"  Class model  : {args.cls_model}")
    print(f"  Output       : {args.output}")
    print(f"  Files        : {len(sources):,}")
    print("=" * 90)

    det_model = YOLO(str(args.detect_model))
    cls_model = YOLO(str(args.cls_model))
    rows: list[dict[str, str]] = []
    started_at = time.perf_counter()

    for source in tqdm(sources, desc="Files"):
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append(process_image(source, det_model, cls_model, args))
        elif source.suffix.lower() in VIDEO_EXTENSIONS:
            rows.append(process_video(source, det_model, cls_model, args))

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["source", "output", "type", "label", "detections", "normal", "sleep"],
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - started_at
    counts = Counter(row["label"] for row in rows)
    print("=" * 90)
    print(" HOAN TAT ".center(90, "="))
    for label in sorted(TARGET_LABELS):
        print(f"  {label:<10}: {counts[label]:>6,} file")
    print(f"  CSV       : {csv_path}")
    print(f"  Output    : {args.output}")
    print(f"  Thoi gian : {elapsed:.2f} giay")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
