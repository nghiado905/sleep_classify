import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "human_detection_2class.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "test-img.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "predict_results"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def predict(
    model_path: Path,
    source: Path,
    output_dir: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "",
):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading 2-class Human Detection model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Number of classes: {len(model.names)}")
    print(f"Classes: {model.names}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect source files
    if source.is_dir():
        image_files = sorted(
            [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
    elif source.suffix.lower() in IMAGE_EXTENSIONS:
        image_files = [source]
    elif source.suffix.lower() in VIDEO_EXTENSIONS:
        # Run prediction directly via Ultralytics for video
        print(f"Processing video: {source}")
        model.predict(
            source=str(source),
            save=True,
            project=str(output_dir),
            name="video_result",
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device if device else None,
        )
        print(f"Saved video prediction to: {output_dir / 'video_result'}")
        return
    else:
        raise ValueError(f"Unsupported source format: {source}")

    if not image_files:
        print(f"No images found in: {source}")
        return

    print(f"Running prediction on {len(image_files)} image(s)...")

    class_counts = {name: 0 for name in model.names.values()}
    total_detections = 0

    for img_path in tqdm(image_files, desc="Detecting"):
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device if device else None,
            verbose=False,
        )
        result = results[0]

        # Statistics
        if result.boxes is not None:
            for cls_id in result.boxes.cls.tolist():
                name = model.names[int(cls_id)]
                class_counts[name] += 1
                total_detections += 1

        # Draw bboxes & save
        annotated_frame = result.plot()
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated_frame)

    print("\n" + "=" * 50)
    print(f"Prediction completed! Saved results to:\n  {output_dir}")
    print("\nDetection Summary:")
    print(f"  Total detections: {total_detections}")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count} boxes")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Detection Predict (2 classes: head, visible-person)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to .pt model")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="Path to image, folder, or video")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', etc.")

    args = parser.parse_args()
    predict(
        model_path=Path(args.model),
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
