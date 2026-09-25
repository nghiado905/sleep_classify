"""Build diverse datasets by filtering similar frames with MobileNet embeddings.

The filter works at frame level. A selected frame always keeps every detected
person, so its YOLO label file remains complete.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, deque
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create complete YOLO frames while filtering visually similar frames."
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
    parser.add_argument("--raisehand-positive-id", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.985,
        help="Skip frame when cosine similarity is at least this value.",
    )
    parser.add_argument(
        "--feature-history", type=int, default=100,
        help="Number of recently saved frames used for comparison.",
    )
    parser.add_argument(
        "--uncertain-confidence", type=float, default=0.7,
        help="Always save a frame containing a prediction below this confidence.",
    )
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


def extract_frame_feature(
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


def layout_signature(
    detections: list[dict], width: int, height: int
) -> tuple[tuple[str, int, int, int, int], ...]:
    """Describe labels and coarse locations so class/layout changes are preserved."""
    signature = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        signature.append((
            detection["label"],
            round(((x1 + x2) / 2) / width * 20),
            round(((y1 + y2) / 2) / height * 20),
            round((x2 - x1) / width * 20),
            round((y2 - y1) / height * 20),
        ))
    return tuple(sorted(signature))


def nearest_similarity(
    feature: np.ndarray,
    signature: tuple,
    history: deque[tuple[np.ndarray, tuple]],
) -> float | None:
    similarities = [
        float(np.dot(feature, old_feature))
        for old_feature, old_signature in history
        if old_signature == signature
    ]
    return max(similarities) if similarities else None


def run(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    if args.frame_stride < 1 or args.feature_history < 1:
        print("[ERROR] frame-stride and feature-history must be >= 1", file=sys.stderr)
        return 1
    if not 0 <= args.similarity_threshold <= 1:
        print("[ERROR] similarity-threshold must be between 0 and 1", file=sys.stderr)
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
    feature_device = resolve_torch_device(args.device)
    if feature_device.type == "cuda" and not torch.cuda.is_available():
        print("[ERROR] CUDA was requested but PyTorch cannot access a GPU.", file=sys.stderr)
        return 1
    feature_model, feature_preprocess = load_feature_model(feature_device)
    visible_ids = resolve_visible_ids(det_model.names, args.visible_class)
    if not visible_ids:
        print(f"[ERROR] Visible class not found in {det_model.names}", file=sys.stderr)
        return 1

    histories: dict[Path, deque[tuple[np.ndarray, tuple]]] = {}
    counters, skipped, rows = Counter(), Counter(), []
    started_at = time.perf_counter()

    for source_path, frame_index, _, image in tqdm(
        iter_frames(inputs, args.frame_stride), desc="Frames"
    ):
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
            if raise_id == args.raisehand_positive_id:
                label, label_confidence = "raisehand", raise_conf
            elif sleep_id == args.sleep_positive_id:
                label, label_confidence = "sleep", sleep_conf
            else:
                label, label_confidence = "normal", min(sleep_conf, raise_conf)
            detections.append({
                "box_index": box_index, "box": box, "crop": crop, "label": label,
                "label_confidence": label_confidence, "sleep_id": sleep_id,
                "sleep_conf": sleep_conf, "raise_id": raise_id, "raise_conf": raise_conf,
            })

        if not detections:
            skipped["no_visible_person"] += 1
            continue

        feature = extract_frame_feature(
            image, feature_model, feature_preprocess, feature_device
        )
        signature = layout_signature(detections, width, height)
        history = histories.setdefault(
            source_path, deque(maxlen=args.feature_history)
        )
        similarity = nearest_similarity(feature, signature, history)
        uncertain = any(
            item["label_confidence"] < args.uncertain_confidence for item in detections
        )
        if similarity is not None and similarity >= args.similarity_threshold and not uncertain:
            skipped["similar_frame"] += 1
            continue

        reason = (
            "uncertain" if uncertain
            else "new_layout" if similarity is None
            else "visual_change"
        )
        history.append((feature, signature))
        frame_stem = (
            source_path.stem if source_path.suffix.lower() in IMAGE_EXTENSIONS
            else f"{source_path.stem}_{frame_index:08d}"
        )
        image_output = unique_path(yolo_root / "images", frame_stem, ".jpg")
        label_output = yolo_root / "labels" / f"{image_output.stem}.txt"
        visualized = image.copy() if args.visualize else None
        lines = []

        for item in detections:
            box_index, box, crop = item["box_index"], item["box"], item["crop"]
            label = item["label"]
            class_id = CLASS_TO_ID[label]
            line = yolo_line(class_id, box, width, height)
            lines.append(line)
            crop_path = unique_path(
                cls_root / label, f"{frame_stem}_box{box_index:03d}_{label}", ".jpg"
            )
            cv2.imwrite(str(crop_path), crop)
            if visualized is not None:
                draw_prediction(visualized, box, label, item["label_confidence"])
            counters[label] += 1
            x1, y1, x2, y2 = box
            rows.append({
                "source": str(source_path), "frame_index": frame_index,
                "sample_reason": reason, "nearest_similarity": "" if similarity is None
                else f"{similarity:.6f}", "yolo_image": str(image_output),
                "yolo_label": str(label_output), "crop": str(crop_path),
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

    fieldnames = [
        "source", "frame_index", "sample_reason", "nearest_similarity",
        "yolo_image", "yolo_label", "crop", "label", "class_id",
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
    print(f"Skipped similar frames: {skipped['similar_frame']:,}")
    print(f"Output: {output}")
    print(f"Time: {time.perf_counter() - started_at:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
