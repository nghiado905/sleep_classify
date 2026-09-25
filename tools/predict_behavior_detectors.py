"""Run drowsiness and hand-raising detectors on images or videos."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from build_detect_classify_datasets import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    collect_inputs,
    iter_frames,
    unique_path,
    yolo_line,
)
from yolov5_drowsiness import YoloV5DrowsinessDetector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLEEP_MODEL = ROOT / "videos" / "drowsiness_yolov5_best.onnx"
DEFAULT_RAISE_MODEL = ROOT / "classroom_behavior_yolov8x_fold0_best.pt"
COLORS = {"normal": (60, 180, 75), "sleep": (30, 80, 230), "raisehand": (0, 200, 255)}
TARGET_IDS = {"normal": 0, "sleep": 1, "raisehand": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sleep and raise-hand boxes, crops, CSV and YOLO labels."
    )
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--sleep-model", type=Path, default=DEFAULT_SLEEP_MODEL)
    parser.add_argument("--raisehand-model", type=Path, default=DEFAULT_RAISE_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "behavior_predict")
    parser.add_argument("--device", default=None, help="Ultralytics device: 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sleep-conf", type=float, default=0.5)
    parser.add_argument("--raisehand-conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--exclude-normal", action="store_true",
        help="Do not save normal/reading/writing detections.",
    )
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True,
        help="Save annotated videos (default: enabled).",
    )
    return parser.parse_args()


def draw(image, box, label: str, raw_name: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    color = COLORS[label]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} [{raw_name}] {confidence:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(image, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 6, y1), color, -1)
    cv2.putText(image, text, (x1 + 3, max(text_h, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    inputs = []
    for source in args.sources:
        inputs.extend(collect_inputs(source.resolve()))
    inputs = list(dict.fromkeys(inputs))
    if not inputs:
        raise SystemExit("No input images or videos found.")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be at least 1.")

    output = args.output.resolve()
    crops_root = output / "crops"
    frames_root = output / "frames"
    labels_root = output / "labels"
    videos_root = output / "videos"
    for label in TARGET_IDS:
        (crops_root / label).mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        videos_root.mkdir(parents=True, exist_ok=True)

    sleep_model = YoloV5DrowsinessDetector(args.sleep_model.resolve(), args.device, args.imgsz)
    raise_model = YOLO(str(args.raisehand_model.resolve()))
    print(f"Sleep model: {args.sleep_model.resolve()}")
    print(f"Raise-hand model: {args.raisehand_model.resolve()} | names={raise_model.names}")

    rows = []
    counts = Counter()
    writers: dict[Path, cv2.VideoWriter] = {}
    for source, frame_index, fps, image in tqdm(
        iter_frames(inputs, args.frame_stride), desc="Predict"
    ):
        height, width = image.shape[:2]
        stem = source.stem if source.suffix.lower() in IMAGE_EXTENSIONS else f"{source.stem}_{frame_index:08d}"
        detections = []

        for prediction in sleep_model.predict_all(image, args.sleep_conf, args.iou):
            if prediction.class_id == 0 and args.exclude_normal:
                continue
            label = "normal" if prediction.class_id == 0 else "sleep"
            detections.append(("drowsiness", prediction.class_id, prediction.class_name,
                               prediction.confidence, label, prediction.box))

        result = raise_model.predict(
            source=image, imgsz=args.imgsz, conf=args.raisehand_conf, iou=args.iou,
            device=args.device, verbose=False,
        )[0]
        if result.boxes is not None:
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                clipped = (
                    max(0, int(round(box[0]))), max(0, int(round(box[1]))),
                    min(width, int(round(box[2]))), min(height, int(round(box[3]))),
                )
                if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                    label = "raisehand" if class_id == 0 else "normal"
                    if label == "normal" and args.exclude_normal:
                        continue
                    detections.append(("raisehand", class_id, str(raise_model.names[class_id]),
                                       float(confidence), label, clipped))

        annotated = image.copy()
        yolo_lines = []
        for index, (model_name, raw_id, raw_name, confidence, label, box) in enumerate(detections, 1):
            class_id = TARGET_IDS[label]
            crop_path = unique_path(crops_root / label, f"{stem}_{index:03d}_{label}", ".jpg")
            x1, y1, x2, y2 = box
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            yolo_lines.append(yolo_line(class_id, box, width, height))
            draw(annotated, box, label, raw_name, confidence)
            counts[label] += 1
            rows.append({
                "source": str(source), "frame_index": frame_index, "model": model_name,
                "raw_class_id": raw_id, "raw_class_name": raw_name,
                "label": label, "class_id": class_id, "confidence": f"{confidence:.6f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop": str(crop_path),
            })

        frame_path = frames_root / f"{stem}.jpg"
        label_path = labels_root / f"{stem}.txt"
        cv2.imwrite(str(frame_path), annotated)
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        if args.save_video and source.suffix.lower() in VIDEO_EXTENSIONS:
            writer = writers.get(source)
            if writer is None:
                video_path = videos_root / f"{source.stem}_pred.mp4"
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1.0, fps / args.frame_stride), (width, height))
                writers[source] = writer
            writer.write(annotated)

    for writer in writers.values():
        writer.release()
    fields = ["source", "frame_index", "model", "raw_class_id", "raw_class_name",
              "label", "class_id", "confidence", "x1", "y1", "x2", "y2", "crop"]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE: normal={counts['normal']}, sleep={counts['sleep']}, raisehand={counts['raisehand']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
