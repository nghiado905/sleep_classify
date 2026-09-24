"""Lệnh huấn luyện mô hình YOLO phân loại sleep/normal."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from training.classifier import train_classifier


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO phân loại ngủ/tỉnh.")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "data_thsc_leloi",
        help="Thư mục nguồn chứa hai thư mục normal/ và sleep/",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "split_leloi",
        help="Dataset đã chia sẵn theo cấu trúc train/val/test",
    )
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Chia lại dữ liệu từ --source và ghi đè --dataset trước khi train",
    )
    parser.add_argument("--model", default="yolo11m-cls.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, 0,1")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--name", default="clean_train")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default="sleep-classification")
    return parser.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args()
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment
    os.environ["MLFLOW_RUN"] = args.name
    train_classifier(
        source_dir=args.source,
        dataset_dir=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        patience=args.patience,
        device=args.device,
        project_dir=args.project,
        run_name=args.name,
        prepare_dataset=args.prepare_dataset,
    )


if __name__ == "__main__":
    main()
