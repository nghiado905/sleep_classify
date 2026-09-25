"""Build one YOLO dataset and one crop-classification dataset."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECT_MODEL = PROJECT_ROOT / "human_detection_2class.pt"
DEFAULT_SLEEP_MODEL = PROJECT_ROOT / "runs" / "runs" / "model_ngu_gat" / "weights" / "best.pt"
DEFAULT_RAISEHAND_MODEL = PROJECT_ROOT / "classify_raisehand.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "detect_classify_auto"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
CLASS_TO_ID = {"normal": 0, "sleep": 1, "raisehand": 2}
CLASS_COLORS = {
    "normal": (60, 180, 75),
    "sleep": (30, 80, 230),
    "raisehand": (0, 200, 255),
}


@dataclass
class TrackState:
    box: tuple[int, int, int, int]
    label: str
    last_seen_frame: int
    last_saved_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect visible people and create dataset_yolo + dataset_cls."
    )
    parser.add_argument("source", type=Path, help="Video/image file or folder")
    parser.add_argument("--detect-model", type=Path, default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--sleep-model", type=Path, default=DEFAULT_SLEEP_MODEL)
    parser.add_argument("--raisehand-model", type=Path, default=DEFAULT_RAISEHAND_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None, help="Example: cpu or 0")
    parser.add_argument("--visible-class", default="visible person")
    parser.add_argument("--sleep-positive-id", type=int, default=1)
    parser.add_argument("--raisehand-positive-id", type=int, default=2)
    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="Process every Nth video frame. Default: 1",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save annotated frames with bounding boxes and predicted labels.",
    )
    parser.add_argument(
        "--adaptive-sampling", action="store_true",
        help="Save stable normal/sleep/raisehand crops at different time intervals.",
    )
    parser.add_argument("--normal-seconds", type=float, default=5.0)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--raisehand-seconds", type=float, default=0.5)
    parser.add_argument("--uncertain-confidence", type=float, default=0.7)
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def collect_inputs(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS else []
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        )
    return []


def iter_frames(
    inputs: list[Path], frame_stride: int
) -> Iterator[tuple[Path, int, float, np.ndarray]]:
    for path in inputs:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image = cv2.imread(str(path))
            if image is not None:
                yield path, 0, 1.0, image
            continue

        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        frame_index = 0
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                yield path, frame_index, fps, frame
            frame_index += 1
        capture.release()


def resolve_visible_ids(names: dict[int, str], visible_class: str) -> set[int]:
    value = visible_class.strip().lower()
    if value.isdigit():
        return {int(value)}

    # Treat "visible person", "visible-person", and "visible_person" as one name.
    normalized_value = re.sub(r"[\s_-]+", "", value)
    return {
        key
        for key, name in names.items()
        if re.sub(r"[\s_-]+", "", str(name).strip().lower()) == normalized_value
    }


def clip_xyxy(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(value)) for value in box[:4]]
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def yolo_line(class_id: int, box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (
        f"{class_id} {((x1 + x2) / 2) / width:.6f} "
        f"{((y1 + y2) / 2) / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def find_track(
    tracks: list[TrackState],
    box: tuple[int, int, int, int],
    frame_index: int,
    max_age_frames: int,
    iou_threshold: float,
    used_track_ids: set[int],
) -> TrackState | None:
    candidates = [
        (box_iou(track.box, box), track)
        for track in tracks
        if frame_index - track.last_seen_frame <= max_age_frames
        and id(track) not in used_track_ids
    ]
    if not candidates:
        return None
    best_iou, best_track = max(candidates, key=lambda item: item[0])
    return best_track if best_iou >= iou_threshold else None


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index:05d}{suffix}"
        index += 1
    return candidate


def prepare_output(output: Path, skip_existing: bool, visualize: bool) -> tuple[Path, Path]:
    if output.exists() and not skip_existing:
        shutil.rmtree(output)
    yolo_root, cls_root = output / "dataset_yolo", output / "dataset_cls"
    for path in (
        yolo_root / "images", yolo_root / "labels", cls_root / "normal",
        cls_root / "sleep", cls_root / "raisehand",
    ):
        path.mkdir(parents=True, exist_ok=True)
    if visualize:
        (output / "visualize").mkdir(parents=True, exist_ok=True)
    (yolo_root / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\n\nnames:\n"
        "  0: normal\n  1: sleep\n  2: raisehand\n",
        encoding="utf-8",
    )
    return yolo_root, cls_root


def classify(model, crop: np.ndarray, imgsz: int, device: str | None) -> tuple[int, float]:
    result = model.predict(source=crop, imgsz=imgsz, device=device, verbose=False)[0]
    if result.probs is None:
        raise RuntimeError("Classification model did not return probabilities.")
    return int(result.probs.top1), float(result.probs.top1conf)


def draw_prediction(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    x1, y1, x2, y2 = box
    color = CLASS_COLORS[label]
    text = f"{label} {confidence:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    text_top = max(0, y1 - text_height - baseline - 6)
    cv2.rectangle(
        image, (x1, text_top), (x1 + text_width + 8, text_top + text_height + baseline + 6),
        color, -1,
    )
    cv2.putText(
        image, text, (x1 + 4, text_top + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    if args.frame_stride < 1:
        print("[ERROR] --frame-stride must be >= 1", file=sys.stderr)
        return 1
    if min(args.normal_seconds, args.sleep_seconds, args.raisehand_seconds) <= 0:
        print("[ERROR] Adaptive sampling intervals must be > 0", file=sys.stderr)
        return 1
    if not 0 <= args.uncertain_confidence <= 1 or not 0 <= args.track_iou <= 1:
        print("[ERROR] Confidence and IoU values must be between 0 and 1", file=sys.stderr)
        return 1
    source, output = args.source.resolve(), args.output.resolve()
    model_paths = {
        "detect": args.detect_model.resolve(), "sleep": args.sleep_model.resolve(),
        "raisehand": args.raisehand_model.resolve(),
    }
    for model_path in model_paths.values():
        if not model_path.is_file():
            print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
            return 1
    inputs = collect_inputs(source)
    if not inputs:
        print(f"[ERROR] No valid videos or images found in: {source}", file=sys.stderr)
        return 1

    yolo_root, cls_root = prepare_output(output, args.skip_existing, args.visualize)
    det_model = YOLO(str(model_paths["detect"]))
    sleep_model = YOLO(str(model_paths["sleep"]))
    raisehand_model = YOLO(str(model_paths["raisehand"]))
    visible_ids = resolve_visible_ids(det_model.names, args.visible_class)
    if not visible_ids:
        print(f"[ERROR] Visible class not found in {det_model.names}", file=sys.stderr)
        return 1

    counters, skipped, rows = Counter(), Counter(), []
    tracks_by_source: dict[Path, list[TrackState]] = {}
    sample_seconds = {
        "normal": args.normal_seconds,
        "sleep": args.sleep_seconds,
        "raisehand": args.raisehand_seconds,
    }
    started_at = time.perf_counter()
    frames = iter_frames(inputs, args.frame_stride)
    for source_path, frame_index, fps, image in tqdm(frames, desc="Frames"):
        height, width = image.shape[:2]
        result = det_model.predict(
            source=image, imgsz=args.det_imgsz, conf=args.conf, iou=args.iou,
            device=args.device, verbose=False,
        )[0]
        if result.boxes is None:
            skipped["no_boxes"] += 1
            continue

        frame_stem = (
            source_path.stem if source_path.suffix.lower() in IMAGE_EXTENSIONS
            else f"{source_path.stem}_{frame_index:08d}"
        )
        image_output = unique_path(yolo_root / "images", frame_stem, ".jpg")
        label_output = yolo_root / "labels" / f"{image_output.stem}.txt"
        detections = []
        frame_reasons: set[str] = set()
        tracks = tracks_by_source.setdefault(source_path, [])
        max_track_age = max(1, int(round(fps * 3.0)))
        tracks[:] = [
            track for track in tracks
            if frame_index - track.last_seen_frame <= max_track_age
        ]
        used_track_ids: set[int] = set()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        for box_index, (det_class, raw_box) in enumerate(zip(classes, boxes), start=1):
            if det_class not in visible_ids:
                skipped["non_visible_person"] += 1
                continue
            box = clip_xyxy(raw_box, width, height)
            if box is None:
                skipped["empty_box"] += 1
                continue
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                skipped["empty_crop"] += 1
                continue

            sleep_id, sleep_conf = classify(sleep_model, crop, args.cls_imgsz, args.device)
            raise_id, raise_conf = classify(raisehand_model, crop, args.cls_imgsz, args.device)
            if raise_id == args.raisehand_positive_id:
                label = "raisehand"
                label_confidence = raise_conf
            elif sleep_id == args.sleep_positive_id:
                label = "sleep"
                label_confidence = sleep_conf
            else:
                label = "normal"
                label_confidence = min(raise_conf, sleep_conf)

            track = find_track(
                tracks, box, frame_index, max_track_age, args.track_iou, used_track_ids
            )
            label_changed = track is not None and track.label != label
            is_uncertain = label_confidence < args.uncertain_confidence
            if track is None:
                track = TrackState(box, label, frame_index, frame_index)
                tracks.append(track)
                object_should_save = True
                object_reason = "new_person"
            else:
                interval_frames = max(1, int(round(sample_seconds[label] * fps)))
                object_should_save = (
                    not args.adaptive_sampling
                    or label_changed
                    or is_uncertain
                    or frame_index - track.last_saved_frame >= interval_frames
                )
                if label_changed:
                    object_reason = "label_changed"
                elif is_uncertain:
                    object_reason = "uncertain"
                else:
                    object_reason = "interval"
                track.box = box
                track.label = label
                track.last_seen_frame = frame_index
            used_track_ids.add(id(track))

            if not args.adaptive_sampling:
                object_should_save = True
                object_reason = "all_frames"
            if object_should_save:
                frame_reasons.add(object_reason)
            class_id = CLASS_TO_ID[label]
            line = yolo_line(class_id, box, width, height)
            detections.append({
                "box_index": box_index, "box": box, "crop": crop, "track": track,
                "label": label, "label_confidence": label_confidence,
                "class_id": class_id, "line": line, "sleep_id": sleep_id,
                "sleep_conf": sleep_conf, "raise_id": raise_id, "raise_conf": raise_conf,
            })

        if not detections:
            skipped["no_visible_person"] += 1
            continue
        if args.adaptive_sampling and not frame_reasons:
            skipped["adaptive_frame"] += 1
            continue

        sample_reason = "+".join(sorted(frame_reasons)) or "all_frames"
        lines = []
        visualized = image.copy() if args.visualize else None
        for detection in detections:
            box_index = detection["box_index"]
            box = detection["box"]
            crop = detection["crop"]
            track = detection["track"]
            label = detection["label"]
            label_confidence = detection["label_confidence"]
            class_id = detection["class_id"]
            line = detection["line"]
            sleep_id, sleep_conf = detection["sleep_id"], detection["sleep_conf"]
            raise_id, raise_conf = detection["raise_id"], detection["raise_conf"]
            x1, y1, x2, y2 = box

            track.last_saved_frame = frame_index
            lines.append(line)
            crop_path = unique_path(
                cls_root / label, f"{frame_stem}_box{box_index:03d}_{label}", ".jpg"
            )
            cv2.imwrite(str(crop_path), crop)
            if visualized is not None:
                draw_prediction(visualized, box, label, label_confidence)
            counters[label] += 1
            rows.append({
                "source": str(source_path), "frame_index": frame_index,
                "sample_reason": sample_reason,
                "yolo_image": str(image_output),
                "yolo_label": str(label_output), "crop": str(crop_path),
                "label": label, "class_id": class_id, "sleep_raw_id": sleep_id,
                "sleep_confidence": f"{sleep_conf:.6f}", "raisehand_raw_id": raise_id,
                "raisehand_confidence": f"{raise_conf:.6f}", "yolo_line": line,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        cv2.imwrite(str(image_output), image)
        label_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if visualized is not None:
            cv2.imwrite(str(output / "visualize" / image_output.name), visualized)

    fieldnames = [
        "source", "frame_index", "sample_reason", "yolo_image", "yolo_label", "crop",
        "label", "class_id",
        "sleep_raw_id", "sleep_confidence", "raisehand_raw_id",
        "raisehand_confidence", "yolo_line", "x1", "y1", "x2", "y2",
    ]
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"DONE: normal={counters['normal']:,}, sleep={counters['sleep']:,}, "
        f"raisehand={counters['raisehand']:,}, total={sum(counters.values()):,}"
    )
    print(f"Output: {output}")
    if skipped:
        print("Skipped:", dict(skipped))
    print(f"Time: {time.perf_counter() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
