"""Huấn luyện YOLO cho bài toán phân loại sleep/normal."""

from __future__ import annotations

import yaml
import mlflow
from pathlib import Path
from typing import Any

from ultralytics import YOLO, settings

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
    prepare_dataset: bool = False,
) -> Any:
    """Chia dữ liệu, in thống kê và bắt đầu huấn luyện mô hình."""
    settings.update({"mlflow": True})
    if prepare_dataset:
        if source_dir.resolve() == dataset_dir.resolve():
            raise ValueError("--source và --dataset phải là hai thư mục khác nhau")
        summary = split_labelled(source_dir, dataset_dir)
        print("Thống kê sau khi chia dữ liệu:")
        for split_name, counts in summary.items():
            print(
                f"  {split_name:<5}: normal={counts['normal']:,}, "
                f"sleep={counts['sleep']:,}"
            )
    else:
        missing_splits = [
            split_name
            for split_name in ("train", "val")
            if not (dataset_dir / split_name).is_dir()
        ]
        if missing_splits:
            missing = ", ".join(missing_splits)
            raise FileNotFoundError(
                f"Dataset {dataset_dir} thiếu thư mục: {missing}. "
                "Dùng --prepare-dataset nếu cần chia lại từ --source."
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

    # Đoạn mã tự động lấy DVC hash để log vào MLflow
    dvc_tags = {}
    dvc_file = Path("data.dvc")
    if not dvc_file.exists():
        # Thử tìm ở thư mục cha nếu đang chạy script từ thư mục con
        dvc_file = Path("../data.dvc")
        
    if dvc_file.exists():
        try:
            with open(dvc_file, "r", encoding="utf-8") as f:
                dvc_data = yaml.safe_load(f)
                dvc_hash = dvc_data.get("outs", [{}])[0].get("md5")
                if dvc_hash:
                    dvc_tags = {
                        "dvc_dataset_hash": dvc_hash,
                        "dvc_remote": "s3://sleepdataset/v1",
                    }
                    print(f"[INFO] Đã tìm thấy DVC Hash: {dvc_hash} để log vào MLflow")
        except Exception as e:
            print(f"[WARN] Lỗi khi đọc file data.dvc: {e}")

    # Bọc quá trình train trong mlflow.start_run để MLflow YOLO callback tự động nhận diện tag này
    with mlflow.start_run(run_name=run_name):
        if dvc_tags:
            mlflow.set_tags(dvc_tags)
        return YOLO(model_name).train(**train_args)
