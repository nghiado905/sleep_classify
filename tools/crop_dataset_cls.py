"""Crop YOLO bounding boxes into an image-classification dataset.

Input layout (default)::

    datasets/
      sleep/images, sleep/labels
      normal/images, normal/labels

Output layout::

    dataset_cls/
      sleep/   # YOLO class 0
      normal/  # YOLO class 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2


CLASS_NAMES = {0: "sleep", 1: "normal"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOGGER = logging.getLogger("crop_dataset_cls")


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def crop_dataset(input_dir: Path, output_dir: Path) -> None:
    LOGGER.info("Starting crop: input=%s, output=%s", input_dir.resolve(), output_dir.resolve())

    for class_name in CLASS_NAMES.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    saved = {class_id: 0 for class_id in CLASS_NAMES}
    skipped = 0

    # `sleep` and `normal` here are source folders. The class of each crop is
    # always taken from the first value in its YOLO label line.
    for source_dir in (input_dir / "sleep", input_dir / "normal"):
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            LOGGER.warning("Missing images/labels folder in %s", source_dir)
            continue

        image_paths = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        LOGGER.info("Source %s: found %d images", source_dir.name, len(image_paths))

        for image_number, image_path in enumerate(image_paths, start=1):
            if image_number == 1 or image_number % 100 == 0 or image_number == len(image_paths):
                LOGGER.info(
                    "Source %s: processing image %d/%d",
                    source_dir.name,
                    image_number,
                    len(image_paths),
                )

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                LOGGER.warning("Label not found for %s", image_path.name)
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                LOGGER.warning("Cannot read %s", image_path)
                skipped += 1
                continue

            image_height, image_width = image.shape[:2]

            for box_index, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines()
            ):
                parts = line.split()
                if not parts:
                    continue

                if len(parts) != 5:
                    LOGGER.warning("Invalid label at %s:%d", label_path, box_index + 1)
                    skipped += 1
                    continue

                try:
                    class_id = int(parts[0])
                    x_center, y_center, box_width, box_height = map(float, parts[1:])
                except ValueError:
                    LOGGER.warning("Invalid values at %s:%d", label_path, box_index + 1)
                    skipped += 1
                    continue

                if class_id not in CLASS_NAMES:
                    LOGGER.warning(
                        "Unknown class %d at %s:%d", class_id, label_path, box_index + 1
                    )
                    skipped += 1
                    continue

                x1 = max(0, round((x_center - box_width / 2) * image_width))
                y1 = max(0, round((y_center - box_height / 2) * image_height))
                x2 = min(image_width, round((x_center + box_width / 2) * image_width))
                y2 = min(image_height, round((y_center + box_height / 2) * image_height))

                if x2 <= x1 or y2 <= y1:
                    LOGGER.warning("Empty crop at %s:%d", label_path, box_index + 1)
                    skipped += 1
                    continue

                crop = image[y1:y2, x1:x2]
                target_class = CLASS_NAMES[class_id]
                # Include the source folder and box index to guarantee unique names.
                output_name = (
                    f"{source_dir.name}_{image_path.stem}_box{box_index:03d}.jpg"
                )
                output_path = output_dir / target_class / output_name

                if not cv2.imwrite(str(output_path), crop):
                    LOGGER.warning("Cannot save %s", output_path)
                    skipped += 1
                    continue

                saved[class_id] += 1

    LOGGER.info("Done")
    for class_id, class_name in CLASS_NAMES.items():
        LOGGER.info("%s (class %d): %d crops", class_name, class_id, saved[class_id])
    LOGGER.info("Skipped: %d", skipped)
    LOGGER.info("Output: %s", output_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO detection labels into cropped classification images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets"),
        help="Input dataset directory (default: datasets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_cls"),
        help="Output classification directory (default: dataset_cls)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("crop_dataset_cls.log"),
        help="Log file path (default: crop_dataset_cls.log)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_file)
    crop_dataset(args.input, args.output)
