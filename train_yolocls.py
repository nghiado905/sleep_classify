from __future__ import annotations

from multiprocessing import freeze_support

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11m-cls.pt")

    results = model.train(
        data=r"D:\SDS\classfy_sleep\datasets\split_leloi",
        epochs=100,
        imgsz=224,
        batch=32,
        device=0,
        workers=8,
        patience=20,
        project=r"D:\SDS\classfy_sleep\runs\classify_leloi_school",
        name="yolov11m_cls_30k",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mixup=0.15,
        mosaic=0.0,
    )
    results.save(r"D:\SDS\classfy_sleep\runs\classify_leloi_school\yolov11m_cls_30k")


if __name__ == "__main__":
    freeze_support()
    main()
