"""Build complete YOLO data and MobileNet-deduplicated classification crops."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from tqdm import tqdm
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from build_detect_classify_datasets import (
    CLASS_TO_ID,
    DEFAULT_DETECT_MODEL,
    DEFAULT_RAISEHAND_MODEL,
    DEFAULT_SLEEP_MODEL,
    IMAGE_EXTENSIONS,
    classify,
    clip_xyxy,
    collect_inputs,
    draw_prediction,
    iter_frames,
    prepare_output,
    resolve_visible_ids,
    unique_path,
    yolo_line,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "feature_diverse"


@dataclass
class CropTrack:
    box: tuple[int, int, int, int]
    label: str
    last_seen_frame: int
    saved_features: deque[np.ndarray] = field(default_factory=deque)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep complete YOLO frames and filter similar classification crops."
    )
    parser.add_argument(
        "sources", type=Path, nargs="+",
        help="One or more video/image files or folders",
    )
    parser.add_argument("--detect-model", type=Path, default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--sleep-model", type=Path, default=DEFAULT_SLEEP_MODEL)
    parser.add_argument("--raisehand-model", type=Path, default=DEFAULT_RAISEHAND_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None, help="Example: 0 or cpu")
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--visible-class", default="visible person")
    parser.add_argument("--sleep-positive-id", type=int, default=1)
    parser.add_argument(
        "--raisehand-positive-id", type=int, default=1,
        help="Raw positive class ID returned by the raise-hand classifier. Default: 1",
    )
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.985,
        help="Skip a classification crop when cosine similarity is at least this value.",
    )
    parser.add_argument(
        "--feature-history", type=int, default=100,
        help="Number of saved crop features kept for each tracked person.",
    )
    parser.add_argument(
        "--uncertain-confidence", type=float, default=0.7,
        help="Always save a classification crop below this confidence.",
    )
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_torch_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value = str(device).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    first_device = value.split(",", 1)[0]
    if first_device.isdigit():
        return torch.device(f"cuda:{first_device}")
    return torch.device(first_device)


def load_feature_model(device: torch.device):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights).features.to(device).eval()
    return model, weights.transforms()


def extract_crop_feature(
    image: np.ndarray,
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    """Return an L2-normalized MobileNetV3 feature embedding."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    batch = preprocess(tensor).unsqueeze(0).to(device)
    with torch.inference_mode():
        feature_map = model(batch)
        embedding = torch_functional.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        embedding = torch_functional.normalize(embedding, dim=1)
    return embedding[0].cpu().numpy().astype(np.float32)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def find_crop_track(
    tracks: list[CropTrack], box: tuple[int, int, int, int], frame_index: int,
    max_age_frames: int, iou_threshold: float, used_ids: set[int],
) -> CropTrack | None:
    candidates = [
        (box_iou(track.box, box), track)
        for track in tracks
        if frame_index - track.last_seen_frame <= max_age_frames
        and id(track) not in used_ids
    ]
    if not candidates:
        return None
    best_iou, track = max(candidates, key=lambda item: item[0])
    return track if best_iou >= iou_threshold else None


def print_video_summary(source_path: Path, stats: Counter) -> None:
    sleep_raw = {
        int(key.removeprefix("sleep_raw_")): value
        for key, value in stats.items() if key.startswith("sleep_raw_")
    }
    raisehand_raw = {
        int(key.removeprefix("raisehand_raw_")): value
        for key, value in stats.items() if key.startswith("raisehand_raw_")
    }
    tqdm.write("-" * 72)
    tqdm.write(f"VIDEO DONE: {source_path.name}")
    tqdm.write(
        f"  frames processed={stats['frames_processed']:,}, "
        f"YOLO frames saved={stats['yolo_frames_saved']:,}, "
        f"visible-person boxes={stats['visible_boxes']:,}"
    )
    tqdm.write(f"  sleep model raw IDs: {sleep_raw}")
    tqdm.write(f"  raisehand model raw IDs: {raisehand_raw}")
    tqdm.write(
        f"  final labels: normal={stats['label_normal']:,}, "
        f"sleep={stats['label_sleep']:,}, raisehand={stats['label_raisehand']:,}"
    )
    tqdm.write(
        f"  CLS crops saved={stats['cls_saved']:,}, "
        f"similar crops skipped={stats['cls_skipped']:,}"
    )


def run(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    if args.frame_stride < 1 or args.feature_history < 1:
        print("[ERROR] frame-stride and feature-history must be >= 1", file=sys.stderr)
        return 1
    if not 0 <= args.similarity_threshold <= 1 or not 0 <= args.track_iou <= 1:
        print("[ERROR] similarity-threshold and track-iou must be between 0 and 1", file=sys.stderr)
        return 1

    output = args.output.resolve()
    model_paths = [
        args.detect_model.resolve(), args.sleep_model.resolve(),
        args.raisehand_model.resolve(),
    ]
    for model_path in model_paths:
        if not model_path.is_file():
            print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
            return 1
    inputs = []
    seen_inputs: set[Path] = set()
    for source in args.sources:
        resolved_source = source.resolve()
        for input_path in collect_inputs(resolved_source):
            resolved_input = input_path.resolve()
            if resolved_input not in seen_inputs:
                seen_inputs.add(resolved_input)
                inputs.append(resolved_input)
    if not inputs:
        source_list = ", ".join(str(path) for path in args.sources)
        print(f"[ERROR] No valid videos or images found in: {source_list}", file=sys.stderr)
        return 1
    print(f"Found {len(inputs):,} input file(s).")

    yolo_root, cls_root = prepare_output(output, args.skip_existing, args.visualize)
    det_model, sleep_model, raisehand_model = [YOLO(str(path)) for path in model_paths]
    print(f"Detector: {model_paths[0]} | names={det_model.names}")
    print(f"Sleep classifier: {model_paths[1]} | names={sleep_model.names}")
    print(f"Raisehand classifier: {model_paths[2]} | names={raisehand_model.names}")
    raisehand_class_ids = (
        set(raisehand_model.names)
        if isinstance(raisehand_model.names, dict)
        else set(range(len(raisehand_model.names)))
    )
    if args.raisehand_positive_id not in raisehand_class_ids:
        print(
            f"[WARNING] raisehand-positive-id={args.raisehand_positive_id} is not in "
            f"raisehand model names {raisehand_model.names}",
            file=sys.stderr,
        )
    feature_device = resolve_torch_device(args.device)
    if feature_device.type == "cuda" and not torch.cuda.is_available():
        print("[ERROR] CUDA was requested but PyTorch cannot access a GPU.", file=sys.stderr)
        return 1
    feature_model, feature_preprocess = load_feature_model(feature_device)
    visible_ids = resolve_visible_ids(det_model.names, args.visible_class)
    if not visible_ids:
        print(f"[ERROR] Visible class not found in {det_model.names}", file=sys.stderr)
        return 1

    tracks_by_source: dict[Path, list[CropTrack]] = {}
    counters, skipped, rows, frame_rows = Counter(), Counter(), [], []
    current_source: Path | None = None
    video_stats = Counter()
    started_at = time.perf_counter()

    for source_path, frame_index, fps, image in tqdm(
        iter_frames(inputs, args.frame_stride), desc="Frames"
    ):
        if current_source != source_path:
            if current_source is not None:
                print_video_summary(current_source, video_stats)
            current_source = source_path
            video_stats = Counter()
            tqdm.write(f"VIDEO START: {source_path}")
        video_stats["frames_processed"] += 1
        height, width = image.shape[:2]
        result = det_model.predict(
            source=image, imgsz=args.det_imgsz, conf=args.conf, iou=args.iou,
            device=args.device, verbose=False,
        )[0]
        if result.boxes is None:
            skipped["no_boxes"] += 1
            continue

        detections = []
        classes = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        for box_index, (det_class, raw_box) in enumerate(zip(classes, boxes), start=1):
            if det_class not in visible_ids:
                continue
            box = clip_xyxy(raw_box, width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            sleep_id, sleep_conf = classify(sleep_model, crop, args.cls_imgsz, args.device)
            raise_id, raise_conf = classify(raisehand_model, crop, args.cls_imgsz, args.device)
            video_stats[f"sleep_raw_{sleep_id}"] += 1
            video_stats[f"raisehand_raw_{raise_id}"] += 1
            video_stats["visible_boxes"] += 1
            if raise_id == args.raisehand_positive_id:
                label, label_confidence = "raisehand", raise_conf
            elif sleep_id == args.sleep_positive_id:
                label, label_confidence = "sleep", sleep_conf
            else:
                label, label_confidence = "normal", min(sleep_conf, raise_conf)
            video_stats[f"label_{label}"] += 1
            detections.append({
                "box_index": box_index, "box": box, "crop": crop, "label": label,
                "label_confidence": label_confidence, "sleep_id": sleep_id,
                "sleep_conf": sleep_conf, "raise_id": raise_id, "raise_conf": raise_conf,
            })

        if not detections:
            skipped["no_visible_person"] += 1
            continue

        frame_stem = (
            source_path.stem if source_path.suffix.lower() in IMAGE_EXTENSIONS
            else f"{source_path.stem}_{frame_index:08d}"
        )
        image_output = unique_path(yolo_root / "images", frame_stem, ".jpg")
        label_output = yolo_root / "labels" / f"{image_output.stem}.txt"
        visualized = image.copy() if args.visualize else None
        lines = []
        tracks = tracks_by_source.setdefault(source_path, [])
        max_track_age = max(1, int(round(fps * 3.0)))
        tracks[:] = [
            track for track in tracks
            if frame_index - track.last_seen_frame <= max_track_age
        ]
        used_track_ids: set[int] = set()

        for item in detections:
            box_index, box, crop = item["box_index"], item["box"], item["crop"]
            label = item["label"]
            class_id = CLASS_TO_ID[label]
            line = yolo_line(class_id, box, width, height)
            lines.append(line)
            feature = extract_crop_feature(
                crop, feature_model, feature_preprocess, feature_device
            )
            track = find_crop_track(
                tracks, box, frame_index, max_track_age, args.track_iou, used_track_ids
            )
            label_changed = track is not None and track.label != label
            if track is None:
                track = CropTrack(
                    box, label, frame_index, deque(maxlen=args.feature_history)
                )
                tracks.append(track)
                similarity = None
                save_crop = True
                reason = "new_person"
            else:
                if label_changed:
                    track.saved_features.clear()
                similarities = [
                    float(np.dot(feature, old_feature))
                    for old_feature in track.saved_features
                ]
                similarity = max(similarities) if similarities else None
                uncertain = item["label_confidence"] < args.uncertain_confidence
                save_crop = (
                    label_changed
                    or uncertain
                    or similarity is None
                    or similarity < args.similarity_threshold
                )
                reason = (
                    "label_changed" if label_changed
                    else "uncertain" if uncertain
                    else "visual_change" if save_crop
                    else "similar_crop"
                )
                track.box = box
                track.label = label
                track.last_seen_frame = frame_index
            used_track_ids.add(id(track))

            crop_path = None
            if save_crop:
                crop_path = unique_path(
                    cls_root / label, f"{frame_stem}_box{box_index:03d}_{label}", ".jpg"
                )
                cv2.imwrite(str(crop_path), crop)
                track.saved_features.append(feature)
                counters[label] += 1
                video_stats["cls_saved"] += 1
            else:
                skipped["similar_crop"] += 1
                video_stats["cls_skipped"] += 1
            if visualized is not None:
                draw_prediction(visualized, box, label, item["label_confidence"])
            x1, y1, x2, y2 = box
            rows.append({
                "source": str(source_path), "frame_index": frame_index,
                "crop_saved": int(save_crop), "sample_reason": reason,
                "nearest_similarity": "" if similarity is None
                else f"{similarity:.6f}", "yolo_image": str(image_output),
                "yolo_label": str(label_output),
                "crop": "" if crop_path is None else str(crop_path),
                "label": label, "class_id": class_id,
                "sleep_raw_id": item["sleep_id"],
                "sleep_confidence": f"{item['sleep_conf']:.6f}",
                "raisehand_raw_id": item["raise_id"],
                "raisehand_confidence": f"{item['raise_conf']:.6f}",
                "yolo_line": line, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

        cv2.imwrite(str(image_output), image)
        label_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if visualized is not None:
            cv2.imwrite(str(output / "visualize" / image_output.name), visualized)
        frame_label_counts = Counter(item["label"] for item in detections)
        frame_rows.append({
            "source_video": str(source_path),
            "frame_index": frame_index,
            "image_name": image_output.name,
            "image_path": str(image_output),
            "label_name": label_output.name,
            "label_path": str(label_output),
            "object_count": len(detections),
            "class_ids": " ".join(str(CLASS_TO_ID[item["label"]]) for item in detections),
            "class_names": " ".join(item["label"] for item in detections),
            "normal_count": frame_label_counts["normal"],
            "sleep_count": frame_label_counts["sleep"],
            "raisehand_count": frame_label_counts["raisehand"],
        })
        video_stats["yolo_frames_saved"] += 1

    if current_source is not None:
        print_video_summary(current_source, video_stats)

    fieldnames = [
        "source", "frame_index", "crop_saved", "sample_reason", "nearest_similarity",
        "yolo_image", "yolo_label", "crop", "label", "class_id",
        "sleep_raw_id", "sleep_confidence", "raisehand_raw_id",
        "raisehand_confidence", "yolo_line", "x1", "y1", "x2", "y2",
    ]
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frame_fields = [
        "source_video", "frame_index", "image_name", "image_path", "label_name",
        "label_path", "object_count", "class_ids", "class_names", "normal_count",
        "sleep_count", "raisehand_count",
    ]
    with (output / "frames.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    print(
        f"DONE: normal={counters['normal']:,}, sleep={counters['sleep']:,}, "
        f"raisehand={counters['raisehand']:,}, total={sum(counters.values()):,}"
    )
    print(f"Skipped similar classification crops: {skipped['similar_crop']:,}")
    print(f"Output: {output}")
    print(f"Time: {time.perf_counter() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
