from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from .dataset_utils import IMAGE_EXTS, collect_images
except ImportError:
    from dataset_utils import IMAGE_EXTS, collect_images


ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize or letterbox images for sleep classification.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Resize or letterbox one image and save it.")
    image_parser.add_argument("--source", type=Path, required=True)
    image_parser.add_argument("--output", type=Path, required=True)
    image_parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    image_parser.add_argument("--mode", choices=("resize", "letterbox"), default="resize")

    folder_parser = subparsers.add_parser("folder", help="Resize or letterbox all images in a folder.")
    folder_parser.add_argument("--source", type=Path, default=ROOT / "datasets" / "labelled")
    folder_parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "processed")
    folder_parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    folder_parser.add_argument("--mode", choices=("resize", "letterbox"), default="resize")

    return parser.parse_args()


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def resize_square(image_bgr: np.ndarray, imgsz: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(image_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)


def letterbox(
    image_bgr: np.ndarray,
    imgsz: int = IMG_SIZE,
    color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image shape")

    scale = min(imgsz / height, imgsz / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_width = imgsz - new_width
    pad_height = imgsz - new_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )


def preprocess_image_bgr(image_bgr: np.ndarray, imgsz: int = IMG_SIZE, mode: str = "resize") -> np.ndarray:
    if mode == "resize":
        return resize_square(image_bgr, imgsz)
    if mode == "letterbox":
        return letterbox(image_bgr, imgsz)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def to_nchw_float32(image_bgr: np.ndarray, imgsz: int = IMG_SIZE, mode: str = "resize") -> np.ndarray:
    image_bgr = preprocess_image_bgr(image_bgr, imgsz=imgsz, mode=mode)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def save_preprocessed_image(source: Path, output: Path, imgsz: int, mode: str) -> None:
    image = load_image_bgr(source)
    processed = preprocess_image_bgr(image, imgsz=imgsz, mode=mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), processed)


def preprocess_folder(source: Path, output_dir: Path, imgsz: int, mode: str) -> Counter:
    counts: Counter = Counter()
    for image_path in collect_images(source):
        try:
            relative_path = image_path.relative_to(source) if source.is_dir() else Path(image_path.name)
        except ValueError:
            relative_path = Path(image_path.name)
        output_path = output_dir / relative_path
        save_preprocessed_image(image_path, output_path, imgsz=imgsz, mode=mode)
        class_name = image_path.parent.name.lower()
        counts[class_name] += 1
    return counts


def main() -> None:
    args = parse_args()
    if args.command == "image":
        save_preprocessed_image(args.source, args.output, imgsz=args.imgsz, mode=args.mode)
        print(f"Output: {args.output}")
    elif args.command == "folder":
        counts = preprocess_folder(args.source, args.output_dir, imgsz=args.imgsz, mode=args.mode)
        for label, count in sorted(counts.items()):
            print(f"{label}: {count}")
        print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
