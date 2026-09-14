"""Huấn luyện YOLO cho bài toán phân loại sleep/normal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from preprocess.dataset_utils import split_labelled


def train_classifier(
    source_dir: Path,
    dataset_dir: Path,
    model_name: str = "yolo11m-cls.pt",
    epochs: int = 200,
    image_size: int = 224,
    batch_size: int = 32,
    patience: int = 20,
    device: str | None = None,
    project_dir: Path | None = None,
    run_name: str = "clean_train",
) -> Any:
    """Chia dữ liệu, in thống kê và bắt đầu huấn luyện mô hình."""
    summary = split_labelled(source_dir, dataset_dir)
    print("Thống kê sau khi chia dữ liệu:")
    for split_name, counts in summary.items():
        print(
            f"  {split_name:<5}: normal={counts['normal']:,}, "
            f"sleep={counts['sleep']:,}"
        )

    train_args: dict[str, object] = {
        "data": str(dataset_dir),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "patience": patience,
        "name": run_name,
    }
    if project_dir is not None:
        train_args["project"] = str(project_dir)
    if device is not None:
        train_args["device"] = device

    return YOLO(model_name).train(**train_args)
