import argparse
import os
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "data.yaml"
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "runs"


def train(
    data_yaml: Path,
    model_name_or_path: str = DEFAULT_MODEL,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "",
    workers: int = 4,
    patience: int = 30,
    project_dir: Path = DEFAULT_PROJECT_DIR,
    run_name: str = "human_train",
):
    if not data_yaml.exists():
        print(f"Error: Dataset config file not found at: {data_yaml}")
        print("Please copy 'data.yaml.template' to 'data.yaml' and update dataset paths.")
        return

    # Auto-detect device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Starting training:")
    print(f"  - Model: {model_name_or_path}")
    print(f"  - Data YAML: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {imgsz}")
    print(f"  - Device: {device}")
    print(f"  - Runs Directory: {project_dir / run_name}")

    # Load model
    model = YOLO(model_name_or_path)

    # Train
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=workers,
        patience=patience,
        project=str(project_dir),
        name=run_name,
        save=True,
        exist_ok=True,
        verbose=True,
    )
    print(f"\nTraining completed! Weights saved under: {project_dir / run_name / 'weights'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Human Detection model (2 classes: head, visible-person)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Base model weight or yaml (e.g. yolo11n.pt, or models/human_detection_2class.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device: '0', '0,1', or 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--project", type=str, default=str(DEFAULT_PROJECT_DIR), help="Runs save directory")
    parser.add_argument("--name", type=str, default="human_train", help="Run name")

    args = parser.parse_args()
    train(
        data_yaml=Path(args.data),
        model_name_or_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project_dir=Path(args.project),
        run_name=args.name,
    )
