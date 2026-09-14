"""Tải và gộp dữ liệu detection Roboflow thành hai lớp YOLO."""

from __future__ import annotations

import argparse
import ast
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from configs.roboflow_sources import (
    DEFAULT_SEED,
    DEFAULT_TARGET_PER_CLASS,
    ROBOFLOW_FORMAT,
    ROBOFLOW_SOURCES,
    SLEEP_CLASS_ALIASES,
    RoboflowSource,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class Sample:
    image: Path
    labels: tuple[tuple[int, float, float, float, float], ...]
    class_name: str
    source_slug: str


def read_class_names(yaml_path: Path) -> list[str]:
    """Read either inline or list-style `names` from an Ultralytics data.yaml."""
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            value = stripped.split(":", 1)[1].strip()
            if value.startswith("["):
                parsed = ast.literal_eval(value)
                return [str(name).strip().lower() for name in parsed]

    names: list[str] = []
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names and stripped.startswith("-"):
            names.append(stripped[1:].strip().strip("'\"").lower())
        elif in_names and stripped and not stripped.startswith("#"):
            break
    return names


def is_sleep_class(name: str) -> bool:
    normalized = name.lower().strip()
    return normalized in SLEEP_CLASS_ALIASES or any(
        alias in normalized for alias in SLEEP_CLASS_ALIASES
    )


def find_matching_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def collect_samples(dataset_path: Path, source_slug: str) -> list[Sample]:
    names = read_class_names(dataset_path / "data.yaml")
    sleep_ids = {index for index, name in enumerate(names) if is_sleep_class(name)}
    normal_ids = set(range(len(names))) - sleep_ids
    print(f"  Classes    : {names}")
    print(f"  ID lớp ngủ : {sorted(sleep_ids)}")
    print(f"  ID lớp tỉnh: {sorted(normal_ids)}")

    samples: list[Sample] = []
    for split in ("train", "valid", "val", "test"):
        labels_dir = dataset_path / split / "labels"
        images_dir = dataset_path / split / "images"
        if not labels_dir.is_dir() or not images_dir.is_dir():
            continue

        for label_path in labels_dir.glob("*.txt"):
            sleep_boxes = []
            normal_boxes = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    box = tuple(map(float, parts[1:5]))
                except ValueError:
                    continue
                if class_id in sleep_ids:
                    sleep_boxes.append((0, *box))
                elif class_id in normal_ids:
                    normal_boxes.append((1, *box))

            image_path = find_matching_image(images_dir, label_path.stem)
            if image_path is None:
                continue
            # A frame containing any sleep instance is assigned to sleep.
            if sleep_boxes:
                samples.append(Sample(image_path, tuple(sleep_boxes), "sleep", source_slug))
            elif normal_boxes:
                samples.append(Sample(image_path, tuple(normal_boxes), "normal", source_slug))
    return samples


def balance_samples(samples: list[Sample], target: int, rng: random.Random) -> list[Sample]:
    if not samples:
        return []
    if len(samples) >= target:
        return rng.sample(samples, target)
    return samples + rng.choices(samples, k=target - len(samples))


def save_samples(samples: Iterable[Sample], output_dir: Path, class_name: str) -> int:
    images_dir = output_dir / class_name / "images"
    labels_dir = output_dir / class_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for index, sample in enumerate(samples):
        image_name = f"{class_name}_{index:05d}{sample.image.suffix.lower()}"
        shutil.copy2(sample.image, images_dir / image_name)
        label_path = labels_dir / f"{class_name}_{index:05d}.txt"
        with label_path.open("w", encoding="utf-8", newline="\n") as label_file:
            for class_id, x, y, width, height in sample.labels:
                label_file.write(
                    f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n"
                )
        count += 1
    return count


def download_source(api_key: str, source: RoboflowSource) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise RuntimeError("Missing dependency: pip install roboflow") from error
    client = Roboflow(api_key=api_key)
    project = client.workspace(source.workspace).project(source.project)
    dataset = project.version(source.version).download(ROBOFLOW_FORMAT)
    return Path(dataset.location)


def build_dataset(api_key: str, output_dir: Path, target: int, seed: int) -> None:
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise RuntimeError(
            f"Thư mục đầu ra không rỗng: {output_dir}. "
            "Hãy chọn --output mới hoặc chủ động dọn thư mục cũ."
        )
    grouped: dict[str, list[Sample]] = {"sleep": [], "normal": []}
    for position, source in enumerate(ROBOFLOW_SOURCES, start=1):
        print(f"\n[{position}/{len(ROBOFLOW_SOURCES)}] {source.slug}")
        try:
            dataset_path = download_source(api_key, source)
            yaml_path = dataset_path / "data.yaml"
            if not yaml_path.is_file():
                print("  Skip: data.yaml not found")
                continue
            samples = collect_samples(dataset_path, source.slug)
            for sample in samples:
                grouped[sample.class_name].append(sample)
            print(
                "  Tìm thấy   : "
                f"{sum(s.class_name == 'sleep' for s in samples)} sleep, "
                f"{sum(s.class_name == 'normal' for s in samples)} normal"
            )
        except Exception as error:
            print(f"  Error      : {error}")

    rng = random.Random(seed)
    final_sleep = balance_samples(grouped["sleep"], target, rng)
    final_normal = balance_samples(grouped["normal"], target, rng)
    if not final_sleep or not final_normal:
        raise RuntimeError("Could not collect both sleep and normal samples")

    sleep_count = save_samples(final_sleep, output_dir, "sleep")
    normal_count = save_samples(final_normal, output_dir, "normal")
    print(f"\nĐã lưu: {sleep_count} sleep, {normal_count} normal")
    print(f"Đầu ra: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải các nguồn Roboflow và tạo bộ YOLO gồm hai lớp."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "yolo_merged",
    )
    parser.add_argument("--target-per-class", type=int, default=DEFAULT_TARGET_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    # Windows có thể dùng bảng mã cp1252 khiến log tiếng Việt bị lỗi.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. In PowerShell: "
            "$env:ROBOFLOW_API_KEY = 'your_key'"
        )
    build_dataset(api_key, args.output, args.target_per_class, args.seed)


if __name__ == "__main__":
    main()
