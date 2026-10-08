from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUGMENTED_ROOT = ROOT.parent / "dataset" / "merged_left_mid_right_newvid_augmented"
DATASETS = {
    "sleep": Path(r'F:\dataset_sds\dataset_cls\split_v1_cluster_normal'),
    "raise_hand": AUGMENTED_ROOT / "dataset_cls_raise_hand_normal",
}
DEFAULT_MODEL = ROOT / "yolo11n-cls.pt"
EXPECTED_CLASSES = {
    "sleep": {"normal", "sleep"},
    "raise_hand": {"normal", "raise_hand"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 classification model for sleep or raise-hand."
    )
    parser.add_argument(
        "--task", choices=sorted(DATASETS), default="sleep",
        help="Binary classification task to train (default: sleep).",
    )
    parser.add_argument(
        "--data", type=Path,
        help="Optional classification dataset containing train/val/test class folders.",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help=f"YOLO classification checkpoint (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32, help="Use -1 for automatic batch size")
    parser.add_argument("--device", default="0", help="Examples: 0, 0,1, cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--name", help="Run name; defaults to yolo11n_cls_<task>")
    parser.add_argument("--cache", action="store_true", help="Cache training images in RAM")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--erasing", type=float, default=0.25)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--degrees", type=float, default=5.0,
                        help="Random rotation range in degrees (default: +/-5).")
    parser.add_argument("--scale", type=float, default=0.2,
                        help="Random image scale/zoom strength.")
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015,
                        help="Hue augmentation fraction.")
    parser.add_argument("--hsv-s", type=float, default=0.7,
                        help="Saturation augmentation fraction.")
    parser.add_argument("--hsv-v", type=float, default=0.4,
                        help="Brightness/value augmentation fraction.")
    parser.add_argument("--bgr", type=float, default=0.0,
                        help="Probability of BGR channel swap (0 disables it).")
    return parser.parse_args()


def validate_dataset(dataset: Path, task: str) -> None:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    expected = EXPECTED_CLASSES[task]
    for split in ("train", "val"):
        split_dir = dataset / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split_dir}")
        classes = {path.name for path in split_dir.iterdir() if path.is_dir()}
        if classes != expected:
            raise ValueError(
                f"Unexpected classes in {split_dir}: {sorted(classes)}; "
                f"expected {sorted(expected)}"
            )
    test_dir = dataset / "test"
    if test_dir.is_dir():
        classes = {path.name for path in test_dir.iterdir() if path.is_dir()}
        if classes != expected:
            raise ValueError(
                f"Unexpected classes in {test_dir}: {sorted(classes)}; "
                f"expected {sorted(expected)}"
            )


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Install it with: pip install ultralytics"
        ) from exc

    dataset = (args.data or DATASETS[args.task]).resolve()
    validate_dataset(dataset, args.task)
    requested_name = Path(args.name) if args.name else None
    if requested_name and requested_name.is_absolute():
        project = requested_name.parent
        run_name = requested_name.name
    else:
        project = ROOT / "runs" / "classify"
        run_name = args.name or f"yolo11n_cls_{args.task}"

    print(f"Task:    {args.task}")
    print(f"Dataset: {dataset}")
    model_path = args.model.resolve() if args.model.exists() else args.model

    print(f"Model:   {model_path}")
    print(f"Classes: {sorted(EXPECTED_CLASSES[args.task])}")

    model = YOLO(str(model_path))
    model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        project=str(project),
        name=run_name,
        exist_ok=False,
        seed=20260930,
        deterministic=True,
        plots=True,
        save=True,
        cache=args.cache,
        dropout=args.dropout,
        erasing=args.erasing,
        mixup=args.mixup,
        auto_augment="randaugment",
        degrees=args.degrees,
        scale=args.scale,
        translate=args.translate,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        bgr=args.bgr,
        cos_lr=True,
        resume=False,
    )


if __name__ == "__main__":
    main()
