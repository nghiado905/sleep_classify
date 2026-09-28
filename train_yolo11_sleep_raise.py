import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(
    r"D:\SDS\classfy_sleep\sleep_raise_merged_model3_model4"
)
DATA_YAML = DATASET_DIR / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 for sleep and raise-hand detection")
    parser.add_argument("--model", default=r"D:\SDS\classfy_sleep\Student Classroom Behavior Dataset\runs\detect\yolo11n_sleep_raise3\weights\best.pt", help="YOLO11 checkpoint or model YAML")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16, help="Use -1 for automatic batch size")
    parser.add_argument("--device", default="0", help="Examples: 0, 0,1, cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--name", default="yolo11n.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint passed to --model")
    return parser.parse_args()


def write_runtime_yaml() -> Path:
    yaml_path = DATASET_DIR / "data_train.yaml"
    dataset_path = DATASET_DIR.as_posix()
    yaml_path.write_text(
        f"path: '{dataset_path}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: sleep\n"
        "  1: raise_hand\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Install it with: pip install ultralytics"
        ) from exc

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    data_yaml = write_runtime_yaml()
    model = YOLO(args.model)
    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(ROOT / "runs" / "detect"),
        "name": args.name,
        "exist_ok": False,
        "seed": 20260928,
        "deterministic": True,
        "plots": True,
        "save": True,
        "resume": args.resume,
    }
    if args.device is not None:
        train_args["device"] = args.device

    model.train(**train_args)


if __name__ == "__main__":
    main()
