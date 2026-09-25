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
    clip_xyxy,
    collect_inputs,
    iter_frames,
    resolve_visible_ids,
    unique_path,
    yolo_line,
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLEEP_MODEL = ROOT / "runs" / "runs" / "model_ngu_gat" / "weights" / "best.pt"
DEFAULT_RAISE_MODEL = ROOT / "videos" / "classroom_behavior_yolov8x_fold0_best.pt"
DEFAULT_PERSON_MODEL = ROOT / "videos" / "human_detection_2class.pt"
COLORS = {"normal": (60, 180, 75), "sleep": (30, 80, 230), "raisehand": (0, 200, 255)}
TARGET_IDS = {"normal": 0, "sleep": 1, "raisehand": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sleep and raise-hand boxes, crops, CSV and YOLO labels."
    )
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--person-model", type=Path, default=DEFAULT_PERSON_MODEL)
    parser.add_argument(
        "--sleep-model", type=Path, default=DEFAULT_SLEEP_MODEL,
        help="Ultralytics classification model with class 0=normal, 1=sleep.",
    )
    parser.add_argument("--raisehand-model", type=Path, default=DEFAULT_RAISE_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "behavior_predict")
    parser.add_argument(
        "--mode", choices=("all", "person", "sleep", "raisehand"), default="all",
        help="Run all models or test one detector only.",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device: 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sleep-imgsz", type=int, default=224)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--sleep-conf", type=float, default=0.8)
    parser.add_argument("--sleep-positive-id", type=int, default=1)
    parser.add_argument("--raisehand-conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--visible-class", default="visible-person")
    parser.add_argument(
        "--exclude-normal", action="store_true",
        help="Do not save visible-person boxes that have no matched event.",
    )
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True,
        help="Save annotated videos (default: enabled).",
    )
    return parser.parse_args()


def draw(image, box, model_name: str, label: str, raw_name: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    color = COLORS[label]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    tag = {
        "person": "PERSON", "sleep_classifier": "SLEEP-CLS",
        "raisehand": "RAISEHAND",
    }[model_name]
    text = f"[{tag}] {label} [{raw_name}] {confidence:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(image, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 6, y1), color, -1)
    cv2.putText(image, text, (x1 + 3, max(text_h, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def intersection_over_box(event_box, person_box) -> float:
    """Fraction of an event box covered by a person box."""
    ex1, ey1, ex2, ey2 = event_box
    px1, py1, px2, py2 = person_box
    ix1, iy1 = max(ex1, px1), max(ey1, py1)
    ix2, iy2 = min(ex2, px2), min(ey2, py2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    event_area = max(1, (ex2 - ex1) * (ey2 - ey1))
    return intersection / event_area


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
    required_models = {
        "all": (args.person_model, args.sleep_model, args.raisehand_model),
        "person": (args.person_model,),
        "sleep": (args.person_model, args.sleep_model),
        "raisehand": (args.raisehand_model,),
    }[args.mode]
    for model_path in required_models:
        if not model_path.resolve().is_file():
            raise SystemExit(f"Model not found: {model_path.resolve()}")

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

    person_model = (
        YOLO(str(args.person_model.resolve()))
        if args.mode in {"all", "person", "sleep"} else None
    )
    sleep_model = YOLO(str(args.sleep_model.resolve())) if args.mode in {"all", "sleep"} else None
    raise_model = YOLO(str(args.raisehand_model.resolve())) if args.mode in {"all", "raisehand"} else None
    visible_ids = resolve_visible_ids(person_model.names, args.visible_class) if person_model else set()
    if person_model and not visible_ids:
        raise SystemExit(f"Visible class not found in person model names: {person_model.names}")
    print(f"Mode: {args.mode}")
    if person_model:
        print(f"[PERSON] {args.person_model.resolve()} | names={person_model.names}")
    if sleep_model:
        print(f"[SLEEP-CLS] {args.sleep_model.resolve()} | names={sleep_model.names}")
    if raise_model:
        print(f"[RAISEHAND] {args.raisehand_model.resolve()} | names={raise_model.names}")

    rows = []
    counts = Counter()
    writers: dict[Path, cv2.VideoWriter] = {}
    log_file = (output / "predict.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        tqdm.write(message)
        log_file.write(message + "\n")
        log_file.flush()

    current_source: Path | None = None
    source_counts = Counter()
    for source, frame_index, fps, image in tqdm(
        iter_frames(inputs, args.frame_stride), desc="Predict"
    ):
        if source != current_source:
            if current_source is not None:
                log(
                    f"VIDEO DONE: {current_source.name} | frames={source_counts['frames']} "
                    f"normal={source_counts['normal']} sleep={source_counts['sleep']} "
                    f"raisehand={source_counts['raisehand']}"
                )
            current_source = source
            source_counts = Counter()
            log(f"VIDEO START: {source}")
        source_counts["frames"] += 1
        height, width = image.shape[:2]
        stem = source.stem if source.suffix.lower() in IMAGE_EXTENSIONS else f"{source.stem}_{frame_index:08d}"
        events = []
        sleep_test_detections = []

        result = raise_model.predict(
            source=image, imgsz=args.imgsz, conf=args.raisehand_conf, iou=args.iou,
            device=args.device, classes=[0], verbose=False,
        )[0] if raise_model else None
        if result is not None and result.boxes is not None:
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                clipped = (
                    max(0, int(round(box[0]))), max(0, int(round(box[1]))),
                    min(width, int(round(box[2]))), min(height, int(round(box[3]))),
                )
                if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                    events.append(("raisehand", class_id, str(raise_model.names[class_id]),
                                   float(confidence), "raisehand", clipped))
                    log(
                        f"  frame={frame_index} [RAISEHAND] raw={raise_model.names[class_id]} "
                        f"conf={float(confidence):.3f} box={clipped}"
                    )

        person_result = person_model.predict(
            source=image, imgsz=args.imgsz, conf=args.person_conf, iou=args.iou,
            device=args.device, classes=sorted(visible_ids), verbose=False,
        )[0] if person_model else None
        people = []
        if person_result is not None and person_result.boxes is not None:
            for person_index, (raw_person_box, person_confidence) in enumerate(zip(
                person_result.boxes.xyxy.cpu().numpy(),
                person_result.boxes.conf.cpu().numpy(),
            ), 1):
                person_box = clip_xyxy(raw_person_box, width, height)
                if person_box is None:
                    continue
                people.append((person_box, float(person_confidence)))
                log(
                    f"  frame={frame_index} [PERSON] id={len(people)} "
                    f"conf={float(person_confidence):.3f} box={person_box}"
                )

        if sleep_model:
            for person_index, (person_box, _) in enumerate(people, 1):
                x1, y1, x2, y2 = person_box
                crop = image[y1:y2, x1:x2]
                cls_result = sleep_model.predict(
                    source=crop, imgsz=args.sleep_imgsz, device=args.device, verbose=False
                )[0]
                if cls_result.probs is None:
                    raise RuntimeError("Sleep model is not a classification model.")
                raw_id = int(cls_result.probs.top1)
                confidence = float(cls_result.probs.top1conf)
                raw_name = str(sleep_model.names[raw_id])
                log(
                    f"  frame={frame_index} [SLEEP-CLS] person={person_index} "
                    f"raw_id={raw_id} raw={raw_name} conf={confidence:.3f} "
                    f"person_box={person_box}"
                )
                if args.mode == "sleep":
                    test_label = (
                        "sleep"
                        if raw_id == args.sleep_positive_id and confidence >= args.sleep_conf
                        else "normal"
                    )
                    sleep_test_detections.append(
                        ("sleep_classifier", raw_id, raw_name, confidence,
                         test_label, person_box)
                    )
                if raw_id == args.sleep_positive_id and confidence >= args.sleep_conf:
                    events.append(
                        ("sleep_classifier", raw_id, raw_name, confidence, "sleep", person_box)
                    )

        detections = []
        if args.mode == "all":
            events_by_person: dict[int, list[tuple]] = {index: [] for index in range(len(people))}
            for event in events:
                scores = [intersection_over_box(event[5], person[0]) for person in people]
                best_person = max(range(len(scores)), key=scores.__getitem__) if scores else None
                if best_person is not None and scores[best_person] >= 0.5:
                    events_by_person[best_person].append(event)
                    log(
                        f"  frame={frame_index} [MATCH] {event[4]} box={event[5]} "
                        f"-> person={best_person + 1} overlap={scores[best_person]:.3f}"
                    )
                else:
                    log(
                        f"  frame={frame_index} [UNMATCHED] {event[4]} box={event[5]} "
                        f"best_overlap={max(scores, default=0.0):.3f}"
                    )
            for person_index, (person_box, person_confidence) in enumerate(people):
                matched_events = events_by_person[person_index]
                if matched_events:
                    detections.extend(matched_events)
                elif not args.exclude_normal:
                    detections.append(
                        ("person", -1, "visible-person", person_confidence, "normal", person_box)
                    )
        elif args.mode == "person":
            detections = [
                ("person", -1, "visible-person", confidence, "normal", box)
                for box, confidence in people
            ]
        elif args.mode == "sleep":
            detections = sleep_test_detections
        else:
            detections = events

        annotated = image.copy()
        yolo_lines = []
        for index, (model_name, raw_id, raw_name, confidence, label, box) in enumerate(detections, 1):
            class_id = TARGET_IDS[label]
            crop_path = unique_path(crops_root / label, f"{stem}_{index:03d}_{label}", ".jpg")
            x1, y1, x2, y2 = box
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            yolo_lines.append(yolo_line(class_id, box, width, height))
            draw(annotated, box, model_name, label, raw_name, confidence)
            counts[label] += 1
            source_counts[label] += 1
            log(
                f"  frame={frame_index} [OUTPUT:{model_name.upper()}] label={label} class={class_id} "
                f"model={model_name} raw={raw_name} conf={confidence:.3f} box={box}"
            )
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

    if current_source is not None:
        log(
            f"VIDEO DONE: {current_source.name} | frames={source_counts['frames']} "
            f"normal={source_counts['normal']} sleep={source_counts['sleep']} "
            f"raisehand={source_counts['raisehand']}"
        )
    for writer in writers.values():
        writer.release()
    log_file.close()
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
