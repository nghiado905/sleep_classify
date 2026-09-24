from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = r'D:\Sortdreams\sleep-classification\classify_sleep\runs\model_ngu_gat\yolov11m_cls_30k\weights\best.pt'
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "datasets" / "datasets" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "v1"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phan loai anh va sap xep ket qua vao tung thu muc nhan."
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="Anh hoac thu muc anh",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=r'D:\Sortdreams\sleep-classification\classify_sleep\15_21_20_15_22_50_07c0ca29.jpg',
        help=f"Anh hoac thu muc anh (mac dinh: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Duong dan model .pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thu muc ket qua")
    parser.add_argument("--imgsz", type=int, default=224, help="Kich thuoc anh dau vao")
    parser.add_argument("--batch", type=int, default=1, help="So anh trong mot batch")
    parser.add_argument("--device", default=None, help="Vi du: cpu, 0, 0,1")
    args = parser.parse_args()
    if args.source is not None and args.path is not None:
        parser.error("Chi truyen mot trong hai cach: source hoac --path.")
    args.source = args.path or args.source or DEFAULT_SOURCE_DIR
    return args


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def unique_destination(directory: Path, source: Path) -> Path:
    """Tao ten file dich duy nhat de khong ghi de anh da co."""
    destination = directory / source.name
    index = 1
    while destination.exists():
        destination = directory / f"{source.stem}_{index:04d}{source.suffix.lower()}"
        index += 1
    return destination


def shorten(value: str, width: int) -> str:
    return value if len(value) <= width else f"...{value[-(width - 3):]}"


def print_header(model_path: Path, source: Path, output_dir: Path, total: int) -> None:
    line = "=" * 88
    print(f"\n{line}")
    print(" YOLO IMAGE CLASSIFICATION ".center(88, "="))
    print(line)
    print(f"  Model    : {model_path}")
    print(f"  Source   : {source}")
    print(f"  Output   : {output_dir}")
    print(f"  So luong : {total:,} anh")
    print(line)
    print(f"  {'TIEN DO':<13} {'TEN ANH':<38}  {'NHAN':<15}  {'DO TIN CAY':>11}")
    print("-" * 88)


def print_progress(index: int, total: int, image: Path, label: str, confidence: float) -> None:
    filename = shorten(image.name, 38)
    print(
        f"  {index:>5}/{total:<5}  {filename:<38}  "
        f"{label:<15}  {confidence:>10.2%}",
        flush=True,
    )


def run_inference(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    source = args.source.resolve()
    model_path = args.model.resolve()
    output_dir = args.output.resolve()

    if not model_path.is_file():
        print(f"[LOI] Khong tim thay model: {model_path}", file=sys.stderr)
        return 1

    images = find_images(source)
    if not images:
        print(f"[LOI] Khong tim thay anh hop le trong: {source}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    print_header(model_path, source, output_dir, len(images))

    started_at = time.perf_counter()
    counts: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    model = YOLO(str(model_path))
    predict_args = {
        "source": [str(path) for path in images],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "stream": True,
        "verbose": False,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    results = model.predict(**predict_args)
    for index, (image_path, result) in enumerate(zip(images, results), start=1):
        if result.probs is None:
            raise RuntimeError("Model khong phai model classification. Hay dung model *-cls.pt.")

        top_index = int(result.probs.top1)
        label = str(result.names[top_index]).strip().lower()
        confidence = float(result.probs.top1conf)

        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(label_dir, image_path)
        shutil.copy2(image_path, destination)

        counts[label] += 1
        rows.append(
            {
                "source": str(image_path),
                "output": str(destination),
                "label": label,
                "confidence": f"{confidence:.6f}",
            }
        )
        print_progress(index, len(images), image_path, label, confidence)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["source", "output", "label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - started_at
    print("=" * 88)
    print(" HOAN TAT ".center(88, "="))
    for label, count in sorted(counts.items()):
        print(f"  {label:<20}: {count:>6,} anh ({count / len(images):>7.2%})")
    print(f"  Tong cong           : {len(images):>6,} anh")
    print(f"  Thoi gian           : {elapsed:>8.2f} giay")
    print(f"  Toc do trung binh   : {elapsed / len(images):>8.3f} giay/anh")
    print(f"  Bao cao CSV         : {csv_path}")
    print(f"  Thu muc ket qua     : {output_dir}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_inference(parse_args()))
