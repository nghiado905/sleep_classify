"""Build YOLO and classification datasets with one detector per behavior."""

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
from build_raise_sleep_detection_dataset import draw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAISE_MODEL = Path(
    r"D:\SDS\classfy_sleep\Student Classroom Behavior Dataset"
    r"\runs\detect\yolo11n_sleep_raise3\weights\best.pt"
)
DEFAULT_SLEEP_MODEL = Path(
    r"D:\SDS\classfy_sleep\Student Classroom Behavior Dataset"
    r"\runs\detect\yolo11n_sleep_raise4\weights\best.pt"
)
EXPECTED_NAMES = {0: "sleep", 1: "raise_hand"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use raise3 for raise_hand and raise4 for sleep, then merge detections."
    )
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--raise-model", type=Path, default=DEFAULT_RAISE_MODEL)
    parser.add_argument("--sleep-model", type=Path, default=DEFAULT_SLEEP_MODEL)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output" / "raise3_sleep4_ensemble"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--raisehand-conf", type=float, default=0.35)
    parser.add_argument("--sleep-conf", type=float, default=0.5)
    parser.add_argument("--raw-conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-empty", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def check_names(model, role: str) -> None:
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_NAMES:
        raise SystemExit(
            f"Unexpected classes for {role} model: {names}; expected {EXPECTED_NAMES}"
        )


def predict_class(
    model,
    image,
    class_id: int,
    threshold: float,
    args: argparse.Namespace,
    width: int,
    height: int,
    model_role: str,
    log,
):
    started = time.perf_counter()
    result = model.predict(
        source=image,
        classes=[class_id],
        imgsz=args.imgsz,
        conf=args.raw_conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]
    accepted = []
    raw_count = 0
    if result.boxes is not None:
        for raw_box, confidence, detected_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            raw_count += 1
            box = clip_xyxy(raw_box, width, height)
            keep = (
                box is not None
                and detected_id == class_id
                and float(confidence) >= threshold
            )
            log(
                f"  [{model_role} RAW] class_id={detected_id} "
                f"class={EXPECTED_NAMES.get(detected_id, 'unknown')} "
                f"conf={float(confidence):.6f} threshold={threshold:.3f} "
                f"box={box} status={'ACCEPT' if keep else 'REJECT'}"
            )
            if keep:
                accepted.append((class_id, float(confidence), box, model_role))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return accepted, raw_count, elapsed_ms


def main(args: argparse.Namespace) -> int:
    yolo_config = ROOT / "runs" / ".ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    from ultralytics import YOLO

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1")
    thresholds = (args.raw_conf, args.raisehand_conf, args.sleep_conf, args.iou)
    if not all(0 <= value <= 1 for value in thresholds):
        raise SystemExit("Confidence and IoU values must be between 0 and 1")
    if args.raw_conf > min(args.raisehand_conf, args.sleep_conf):
        raise SystemExit("--raw-conf must not exceed per-class thresholds")

    raise_path = args.raise_model.resolve()
    sleep_path = args.sleep_model.resolve()
    for role, path in (("RAISE3", raise_path), ("SLEEP4", sleep_path)):
        if not path.is_file():
            raise SystemExit(f"{role} model not found: {path}")

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

    raise_model = YOLO(str(raise_path))
    sleep_model = YOLO(str(sleep_path))
    check_names(raise_model, "RAISE3")
    check_names(sleep_model, "SLEEP4")
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

    log(f"[RAISE3 MODEL] path={raise_path} class=1 raise_hand")
    log(f"[SLEEP4 MODEL] path={sleep_path} class=0 sleep")
    log(
        f"[CONFIG] inputs={len(inputs)} device={args.device} imgsz={args.imgsz} "
        f"raisehand_conf={args.raisehand_conf} sleep_conf={args.sleep_conf} "
        f"raw_conf={args.raw_conf} iou={args.iou} stride={args.frame_stride}"
    )

    rows = []
    counts = Counter()
    video_counts = Counter()
    current_source = None
    total_started = time.perf_counter()

    for source, frame_index, _, image in tqdm(
        iter_frames(inputs, args.frame_stride), desc="Ensemble detect"
    ):
        if source != current_source:
            if current_source is not None:
                log(f"[VIDEO DONE] {current_source.name} stats={dict(video_counts)}")
            current_source = source
            video_counts = Counter()
            log(f"[VIDEO START] {source}")

        video_counts["frames_processed"] += 1
        height, width = image.shape[:2]
        raise_dets, raise_raw, raise_ms = predict_class(
            raise_model, image, 1, args.raisehand_conf, args,
            width, height, "RAISE3", log,
        )
        sleep_dets, sleep_raw, sleep_ms = predict_class(
            sleep_model, image, 0, args.sleep_conf, args,
            width, height, "SLEEP4", log,
        )
        detections = sleep_dets + raise_dets
        log(
            f"[FRAME] video={source.name} frame={frame_index} "
            f"raise3_raw={raise_raw} raise3_accepted={len(raise_dets)} "
            f"raise3_ms={raise_ms:.2f} sleep4_raw={sleep_raw} "
            f"sleep4_accepted={len(sleep_dets)} sleep4_ms={sleep_ms:.2f} "
            f"merged={len(detections)}"
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

        for detection_index, (class_id, confidence, box, model_role) in enumerate(
            detections, 1
        ):
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
                draw(annotated, box, class_id, f"{name}/{model_role}", confidence)
            counts[name] += 1
            video_counts[name] += 1
            log(
                f"  [SAVED] model={model_role} class_id={class_id} class={name} "
                f"conf={confidence:.6f} box={box} crop_saved={crop_saved} crop={crop_path}"
            )
            rows.append({
                "source": str(source),
                "frame_index": frame_index,
                "image": str(image_path),
                "label_file": str(label_path),
                "model": model_role,
                "class_id": class_id,
                "class_name": name,
                "confidence": f"{confidence:.6f}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop": str(crop_path),
                "yolo_line": line,
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
        "source", "frame_index", "image", "label_file", "model", "class_id",
        "class_name", "confidence", "x1", "y1", "x2", "y2", "crop", "yolo_line",
    ]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log(
        f"[DONE] raise_hand={counts['raise_hand']} sleep={counts['sleep']} "
        f"rows={len(rows)} elapsed={time.perf_counter() - total_started:.2f}s output={output}"
    )
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
