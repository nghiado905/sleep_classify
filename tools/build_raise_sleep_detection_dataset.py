"""Create YOLO frames/labels and classification crops from a two-class detector."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm

from build_detect_classify_datasets import (
    IMAGE_EXTENSIONS,
    clip_xyxy,
    collect_inputs,
    iter_frames,
    unique_path,
    yolo_line,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (r'D:\SDS\classfy_sleep\Student Classroom Behavior Dataset\runs\detect\yolo11n_sleep_raise4\weights\best.pt'
)
EXPECTED_NAMES = {0: "sleep", 1: "raise_hand"}
COLORS = {0: (30, 80, 230), 1: (0, 200, 255)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO and classification datasets from raise-hand/sleep detection."
    )
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "raise_sleep_detect")
    parser.add_argument("--device", default="0", help="GPU index such as 0, or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--raisehand-conf", type=float, default=0.35,
        help="Minimum confidence for class 1 raise_hand (default: 0.35).",
    )
    parser.add_argument(
        "--sleep-conf", type=float, default=0.5,
        help="Minimum confidence for class 0 sleep (default: 0.5).",
    )
    parser.add_argument(
        "--raw-conf", type=float, default=0.05,
        help="YOLO inference threshold before per-class filtering (default: 0.05).",
    )
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--save-empty", action="store_true",
        help="Also save frames without raise_hand/sleep detections.",
    )
    parser.add_argument(
        "--keep-output", action="store_true",
        help="Keep existing output instead of replacing it.",
    )
    return parser.parse_args()


def draw(image, box, class_id: int, name: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    color = COLORS[class_id]
    text = f"{name} {confidence:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(0, y1 - text_h - 8)
    cv2.rectangle(image, (x1, top), (x1 + text_w + 6, y1), color, -1)
    cv2.putText(image, text, (x1 + 3, max(text_h, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main(args: argparse.Namespace) -> int:
    yolo_config = ROOT / "runs" / ".ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    from ultralytics import YOLO

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    if not (
        0 <= args.raw_conf <= 1
        and 0 <= args.raisehand_conf <= 1
        and 0 <= args.sleep_conf <= 1
    ):
        raise SystemExit("Confidence thresholds must be between 0 and 1")
    if args.raw_conf > min(args.raisehand_conf, args.sleep_conf):
        raise SystemExit("--raw-conf must not exceed the per-class thresholds")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    inputs = []
    for source in args.sources:
        inputs.extend(collect_inputs(source.resolve()))
    inputs = list(dict.fromkeys(inputs))
    if not inputs:
        raise SystemExit("No input videos or images found")

    output = args.output.resolve()
    if output.exists() and not args.keep_output:
        shutil.rmtree(output)
    yolo_images = output / "dataset_yolo" / "images"
    yolo_labels = output / "dataset_yolo" / "labels"
    cls_root = output / "dataset_cls"
    visualize_root = output / "visualize"
    for path in (yolo_images, yolo_labels, cls_root / "raise_hand", cls_root / "sleep"):
        path.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        visualize_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_NAMES:
        raise SystemExit(f"Unexpected model classes: {names}; expected {EXPECTED_NAMES}")
    (output / "dataset_yolo" / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\n\nnames:\n"
        "  0: sleep\n  1: raise_hand\n",
        encoding="utf-8",
    )

    log_file = (output / "predict.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='milliseconds')} {message}"
        tqdm.write(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"[MODEL] path={model_path} names={names}")
    log(
        f"[CONFIG] inputs={len(inputs)} device={args.device} imgsz={args.imgsz} "
        f"raisehand_conf={args.raisehand_conf} sleep_conf={args.sleep_conf} "
        f"raw_conf={args.raw_conf} iou={args.iou} frame_stride={args.frame_stride}"
    )
    rows = []
    counts = Counter()
    current_source = None
    video_counts = Counter()
    started = time.perf_counter()

    for source, frame_index, _, image in tqdm(
        iter_frames(inputs, args.frame_stride), desc="Detect"
    ):
        if source != current_source:
            if current_source is not None:
                log(f"[VIDEO DONE] {current_source.name} stats={dict(video_counts)}")
            current_source = source
            video_counts = Counter()
            log(f"[VIDEO START] {source}")
        video_counts["frames_processed"] += 1
        height, width = image.shape[:2]
        inference_started = time.perf_counter()
        result = model.predict(
            source=image, imgsz=args.imgsz,
            conf=args.raw_conf, iou=args.iou,
            device=args.device, verbose=False,
        )[0]
        detections = []
        raw_count = 0
        rejected_count = 0
        if result.boxes is not None:
            for raw_box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                raw_count += 1
                box = clip_xyxy(raw_box, width, height)
                class_threshold = (
                    args.raisehand_conf if class_id == 1 else args.sleep_conf
                )
                accepted = (
                    box is not None
                    and class_id in EXPECTED_NAMES
                    and float(confidence) >= class_threshold
                )
                class_name = names.get(class_id, f"unknown_{class_id}")
                log(
                    f"  [RAW DETECTION] class_id={class_id} class={class_name} "
                    f"conf={float(confidence):.6f} threshold={class_threshold:.3f} "
                    f"box={box} status={'ACCEPT' if accepted else 'REJECT'}"
                )
                if accepted:
                    detections.append((class_id, float(confidence), box))
                else:
                    rejected_count += 1
        log(
            f"[FRAME] video={source.name} frame={frame_index} raw={raw_count} "
            f"accepted={len(detections)} rejected={rejected_count} "
            f"time_ms={(time.perf_counter() - inference_started) * 1000:.2f}"
        )
        if not detections and not args.save_empty:
            video_counts["empty_skipped"] += 1
            continue

        frame_stem = (
            source.stem if source.suffix.lower() in IMAGE_EXTENSIONS
            else f"{source.stem}_{frame_index:08d}"
        )
        image_path = unique_path(yolo_images, frame_stem, ".jpg")
        label_path = yolo_labels / f"{image_path.stem}.txt"
        annotated = image.copy() if args.visualize else None
        yolo_lines = []
        for detection_index, (class_id, confidence, box) in enumerate(detections, 1):
            name = EXPECTED_NAMES[class_id]
            x1, y1, x2, y2 = box
            crop_path = unique_path(
                cls_root / name,
                f"{image_path.stem}_{detection_index:03d}_{name}",
                ".jpg",
            )
            crop_saved = cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            line = yolo_line(class_id, box, width, height)
            yolo_lines.append(line)
            if annotated is not None:
                draw(annotated, box, class_id, name, confidence)
            counts[name] += 1
            video_counts[name] += 1
            log(
                f"  [DETECTION] class_id={class_id} class={name} "
                f"conf={confidence:.6f} box={box} crop_saved={crop_saved} crop={crop_path}"
            )
            rows.append({
                "source": str(source), "frame_index": frame_index,
                "image": str(image_path), "label_file": str(label_path),
                "class_id": class_id, "class_name": name,
                "confidence": f"{confidence:.6f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop": str(crop_path), "yolo_line": line,
            })

        cv2.imwrite(str(image_path), image)
        label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )
        if annotated is not None:
            cv2.imwrite(str(visualize_root / image_path.name), annotated)
        video_counts["frames_saved"] += 1

    if current_source is not None:
        log(f"[VIDEO DONE] {current_source.name} stats={dict(video_counts)}")
    fields = [
        "source", "frame_index", "image", "label_file", "class_id", "class_name",
        "confidence", "x1", "y1", "x2", "y2", "crop", "yolo_line",
    ]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log(
        f"[DONE] raise_hand={counts['raise_hand']} sleep={counts['sleep']} "
        f"rows={len(rows)} elapsed={time.perf_counter() - started:.2f}s output={output}"
    )
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
