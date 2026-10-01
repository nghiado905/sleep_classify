"""Detect visible people, crop them, then classify each crop as sleep or normal."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECT_MODEL = ROOT / "videos" / "human_detection_2class.pt"
DEFAULT_SLEEP_MODEL = (
    r'D:\SDS\classfy_sleep\sleep_classify\runs\classify\yolo11n_cls_sleep_v4_hardnegative2\weights\best.pt'
)
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COLORS = {"normal": (60, 180, 75), "sleep": (30, 80, 230)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect visible people and classify each person as sleep or normal."
    )
    parser.add_argument("source", type=Path, help="Video/image file or folder")
    parser.add_argument("--detect-model", type=Path, default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--sleep-model", type=Path, default=DEFAULT_SLEEP_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "person_sleep")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.7)
    parser.add_argument("--sleep-threshold", type=float, default=0.8)
    parser.add_argument("--sleep-positive-id", type=int, default=None)
    parser.add_argument("--visible-class", default="visible-person")
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--log-predictions", action=argparse.BooleanOptionalAction, default=True,
        help="Log probabilities for every detected person (default: enabled).",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    return re.sub(r"[\s_-]+", "", str(value).strip().lower())


def collect_inputs(source: Path) -> list[Path]:
    extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    if source.is_file():
        return [source] if source.suffix.lower() in extensions else []
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
    return []


def resolve_class_id(names: dict[int, str], requested: str) -> int | None:
    if requested.strip().isdigit():
        return int(requested)
    target = normalize_name(requested)
    return next(
        (int(class_id) for class_id, name in names.items() if normalize_name(name) == target),
        None,
    )


def resolve_sleep_id(names: dict[int, str], configured: int | None) -> int:
    if configured is not None:
        return configured
    for class_id, name in names.items():
        if normalize_name(name) in {"sleep", "sleeping", "drowsy", "drowsiness"}:
            return int(class_id)
    raise ValueError(
        f"Cannot infer sleep class from {names}. Pass --sleep-positive-id explicitly."
    )


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(round(value)) for value in box[:4])
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def draw_box(
    image: np.ndarray, box: tuple[int, int, int, int], label: str,
    sleep_probability: float, detect_confidence: float,
) -> None:
    x1, y1, x2, y2 = box
    color = COLORS[label]
    text = f"{label} sleep={sleep_probability:.2f} person={detect_confidence:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image, text, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 2, cv2.LINE_AA,
    )


def process_input(path: Path, args: argparse.Namespace, detector, classifier, visible_id: int,
                  sleep_id: int) -> list[dict]:
    is_image = path.suffix.lower() in IMAGE_EXTENSIONS
    capture = None if is_image else cv2.VideoCapture(str(path))
    image = cv2.imread(str(path)) if is_image else None
    if is_image and image is None:
        print(f"[WARN] Cannot read image: {path}", file=sys.stderr)
        return []
    if capture is not None and not capture.isOpened():
        print(f"[WARN] Cannot open video: {path}", file=sys.stderr)
        return []

    output_dir = args.output / path.stem
    crops_root = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 if is_image else float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = image.shape[1] if is_image else int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = image.shape[0] if is_image else int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = 1 if is_image else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    if args.save_video and not is_image:
        writer = cv2.VideoWriter(
            str(output_dir / "visualize.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
            fps / args.frame_stride, (width, height),
        )

    rows = []
    frame_index = 0
    progress = tqdm(total=total, desc=path.name, unit="frame")
    while True:
        if is_image:
            frame = image if frame_index == 0 else None
            ok = frame is not None
        else:
            ok, frame = capture.read()
        if not ok:
            break
        progress.update(1)
        if frame_index % args.frame_stride:
            frame_index += 1
            continue

        annotated = frame.copy()
        detection = detector.predict(
            frame, imgsz=args.det_imgsz, conf=args.det_conf, iou=args.det_iou,
            device=args.device, verbose=False,
        )[0]
        person_number = 0
        if detection.boxes is not None:
            classes = detection.boxes.cls.cpu().numpy().astype(int)
            confidences = detection.boxes.conf.cpu().numpy()
            boxes = detection.boxes.xyxy.cpu().numpy()
            for raw_box, class_id, det_conf in zip(boxes, classes, confidences):
                if class_id != visible_id:
                    continue
                box = clip_box(raw_box, frame.shape[1], frame.shape[0])
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                person_number += 1
                result = classifier.predict(
                    crop, imgsz=args.cls_imgsz, device=args.device, verbose=False,
                )[0]
                if result.probs is None:
                    raise RuntimeError("Sleep model did not return classification probabilities")
                probabilities = result.probs.data.cpu().numpy()
                sleep_probability = float(probabilities[sleep_id])
                normal_id = next(
                    (
                        int(class_id) for class_id, name in classifier.names.items()
                        if normalize_name(name) == "normal"
                    ),
                    None,
                )
                normal_probability = (
                    float(probabilities[normal_id])
                    if normal_id is not None
                    else float(max(0.0, 1.0 - sleep_probability))
                )
                class_probabilities = ", ".join(
                    f"{classifier.names[int(class_id)]}={float(probability) * 100:.2f}%"
                    for class_id, probability in enumerate(probabilities)
                )
                label = "sleep" if sleep_probability >= args.sleep_threshold else "normal"
                if args.log_predictions:
                    tqdm.write(
                        f"[PREDICT] video={path.name} frame={frame_index} "
                        f"person={person_number} box=({x1},{y1},{x2},{y2}) "
                        f"detect={float(det_conf) * 100:.2f}% | "
                        f"{class_probabilities} | label={label} "
                        f"(sleep threshold={args.sleep_threshold * 100:.2f}%)"
                    )
                crop_path = ""
                if args.save_crops:
                    crop_dir = crops_root / label
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    destination = crop_dir / f"{path.stem}_{frame_index:08d}_{person_number:03d}.jpg"
                    cv2.imwrite(str(destination), crop)
                    crop_path = str(destination)
                draw_box(annotated, box, label, sleep_probability, float(det_conf))
                rows.append({
                    "source": str(path), "frame_index": frame_index,
                    "person_index": person_number, "label": label,
                    "sleep_probability": f"{sleep_probability:.6f}",
                    "sleep_percent": f"{sleep_probability * 100:.2f}",
                    "normal_probability": f"{normal_probability:.6f}",
                    "normal_percent": f"{normal_probability * 100:.2f}",
                    "class_probabilities": class_probabilities,
                    "detect_confidence": f"{float(det_conf):.6f}",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "crop": crop_path,
                })
        if writer is not None:
            writer.write(annotated)
        elif is_image and args.save_video:
            cv2.imwrite(str(output_dir / "visualize.jpg"), annotated)
        frame_index += 1

    progress.close()
    if capture is not None:
        capture.release()
    if writer is not None:
        writer.release()
    return rows


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    for value, name in (
        (args.det_conf, "--det-conf"), (args.det_iou, "--det-iou"),
        (args.sleep_threshold, "--sleep-threshold"),
    ):
        if not 0 <= value <= 1:
            raise SystemExit(f"{name} must be between 0 and 1")
    inputs = collect_inputs(args.source.resolve())
    if not inputs:
        raise SystemExit(f"No supported input found: {args.source}")
    for model_path in (args.detect_model, args.sleep_model):
        if not model_path.is_file():
            raise SystemExit(f"Model not found: {model_path}")

    from ultralytics import YOLO

    detector = YOLO(str(args.detect_model))
    classifier = YOLO(str(args.sleep_model))
    visible_id = resolve_class_id(detector.names, args.visible_class)
    if visible_id is None:
        raise SystemExit(
            f"Visible class '{args.visible_class}' not found in {detector.names}"
        )
    sleep_id = resolve_sleep_id(classifier.names, args.sleep_positive_id)
    print(f"[CONFIG] detector={args.detect_model} names={detector.names}")
    print(f"[CONFIG] classifier={args.sleep_model} names={classifier.names}")
    print(
        f"[CONFIG] visible_id={visible_id}, sleep_id={sleep_id}, "
        f"sleep_threshold={args.sleep_threshold}, device={args.device}"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(inputs, 1):
        print(f"[VIDEO {index}/{len(inputs)}] {path}")
        rows.extend(process_input(path, args, detector, classifier, visible_id, sleep_id))
    csv_path = args.output / "predictions.csv"
    fields = [
        "source", "frame_index", "person_index", "label", "sleep_probability",
        "sleep_percent", "normal_probability", "normal_percent",
        "class_probabilities", "detect_confidence", "x1", "y1", "x2", "y2",
        "crop",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    sleep_count = sum(row["label"] == "sleep" for row in rows)
    print(
        f"[DONE] people={len(rows):,}, sleep={sleep_count:,}, "
        f"normal={len(rows) - sleep_count:,}, csv={csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
