"""Test only the raise-hand classification model on person crop images."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "videos" / "classify_raisehand.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "test_raisehand"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test classify_raisehand.pt on image crops only."
    )
    parser.add_argument("sources", type=Path, nargs="+", help="Image files or folders")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None, help="Example: 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument(
        "--conf-threshold", type=float, default=0.8,
        help="Minimum confidence for raw class 1 to become raisehand.",
    )
    parser.add_argument("--normal-id", type=int, default=0)
    parser.add_argument("--raisehand-id", type=int, default=1)
    return parser.parse_args()


def collect_images(sources: list[Path]) -> list[Path]:
    images: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        source = source.resolve()
        candidates = [source] if source.is_file() else source.rglob("*") if source.is_dir() else []
        for path in candidates:
            path = path.resolve()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path not in seen:
                seen.add(path)
                images.append(path)
    return sorted(images)


def unique_output_dir(base: Path) -> Path:
    base = base.resolve()
    if not base.exists():
        base.mkdir(parents=True)
        return base
    index = 2
    while True:
        candidate = base.parent / f"{base.name}{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def draw_result(image, text: str, positive: bool) -> None:
    color = (0, 200, 255) if positive else (60, 180, 75)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    )
    cv2.rectangle(image, (0, 0), (text_width + 20, text_height + baseline + 18), color, -1)
    cv2.putText(
        image, text, (10, text_height + 8), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (255, 255, 255), 2, cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    if not 0 <= args.conf_threshold <= 1:
        print("[ERROR] --conf-threshold must be between 0 and 1", file=sys.stderr)
        return 1
    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 1
    images = collect_images(args.sources)
    if not images:
        print("[ERROR] No input images found.", file=sys.stderr)
        return 1

    output = unique_output_dir(args.output)
    visualize_dir = output / "visualize"
    visualize_dir.mkdir()
    model = YOLO(str(model_path))
    print(f"Model: {model_path}")
    print(f"Task: {model.task}")
    print(f"Names: {model.names}")
    print(f"Images: {len(images):,}")

    rows, counts = [], Counter()
    for index, image_path in enumerate(tqdm(images, desc="Raisehand test"), start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            counts["unreadable"] += 1
            continue
        result = model.predict(
            source=image, imgsz=args.imgsz, device=args.device, verbose=False
        )[0]
        if result.probs is None:
            print(f"[ERROR] Model did not return classification probabilities: {image_path}")
            return 1

        raw_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        raw_name = str(
            model.names.get(raw_id, raw_id)
            if isinstance(model.names, dict)
            else model.names[raw_id]
        )
        is_raisehand = raw_id == args.raisehand_id and confidence >= args.conf_threshold
        final_label = "raisehand" if is_raisehand else "normal"
        final_class_id = 2 if is_raisehand else 0
        counts[f"raw_{raw_id}"] += 1
        counts[final_label] += 1

        output_name = f"{index:06d}_{image_path.stem}.jpg"
        visualized = image.copy()
        draw_result(
            visualized,
            f"raw={raw_name} ({raw_id}) conf={confidence:.3f} -> {final_label} ({final_class_id})",
            is_raisehand,
        )
        cv2.imwrite(str(visualize_dir / output_name), visualized)
        rows.append({
            "source": str(image_path),
            "visualize": str(visualize_dir / output_name),
            "raw_class_id": raw_id,
            "raw_class_name": raw_name,
            "confidence": f"{confidence:.6f}",
            "passed_threshold": int(is_raisehand),
            "final_label": final_label,
            "final_class_id": final_class_id,
        })

    fields = [
        "source", "visualize", "raw_class_id", "raw_class_name", "confidence",
        "passed_threshold", "final_label", "final_class_id",
    ]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    raw_counts = {
        int(key.removeprefix("raw_")): value
        for key, value in counts.items() if key.startswith("raw_")
    }
    print("-" * 72)
    print(f"Raw class counts: {raw_counts}")
    print(f"Final: normal={counts['normal']:,}, raisehand={counts['raisehand']:,}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
