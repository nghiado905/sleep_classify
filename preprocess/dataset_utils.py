from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ("normal", "sleep")
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}
RANDOM_SEED = 42
MIN_IMAGE_SIZE = 10


def collect_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def copy_unique(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    for index in range(1, 100000):
        candidate = dst_dir / f"{src.stem}_{index:05d}{src.suffix.lower()}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
    raise RuntimeError(f"Could not find unique output name for {src}")


def is_valid_image(path: Path, min_size: int = MIN_IMAGE_SIZE) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        return False, f"unreadable image: {exc}"

    if width < min_size or height < min_size:
        return False, f"image size ({width}, {height}) <{min_size} pixels"
    return True, ""


def split_labelled(
    input_dir: Path,
    output_dir: Path,
    seed: int = RANDOM_SEED,
    ratios: dict[str, float] = SPLIT_RATIOS,
    min_size: int = MIN_IMAGE_SIZE,
    balance: bool = True,
) -> dict[str, Counter]:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    rng = random.Random(seed)
    summary: dict[str, Counter] = {split: Counter() for split in ratios}
    skipped = Counter()
    balanced_out = Counter()
    images_by_class: dict[str, list[Path]] = {}

    for class_name in CLASSES:
        images = []
        for image_path in collect_images(input_dir / class_name):
            ok, reason = is_valid_image(image_path, min_size=min_size)
            if ok:
                images.append(image_path)
            else:
                skipped[class_name] += 1
                print(f"skip {class_name}: {image_path}: {reason}")

        if not images:
            raise RuntimeError(f"No images found for class: {class_name}")

        rng.shuffle(images)
        images_by_class[class_name] = images

    if balance:
        target_count = min(len(images) for images in images_by_class.values())
        for class_name, images in images_by_class.items():
            balanced_out[class_name] = max(len(images) - target_count, 0)
            images_by_class[class_name] = images[:target_count]

    for class_name, images in images_by_class.items():
        total = len(images)
        train_count = int(total * ratios["train"])
        val_count = int(total * ratios["val"])
        split_map = {
            "train": images[:train_count],
            "val": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:],
        }

        for split_name, split_images in split_map.items():
            out_dir = output_dir / split_name / class_name
            for src in split_images:
                copy_unique(src, out_dir)
            summary[split_name][class_name] = len(split_images)

    if skipped:
        summary["_skipped"] = skipped
    if balanced_out:
        summary["_balanced_out"] = balanced_out
    return summary

