"""OpenCV-DNN inference for the legacy YOLOv5 drowsiness model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DROWSINESS_NAMES = {0: "normal", 1: "drowsy", 2: "drowsy#2", 3: "yawning"}


@dataclass(frozen=True)
class DrowsinessPrediction:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int] | None = None


class YoloV5DrowsinessDetector:
    def __init__(self, model_path: Path, device: str | None = None, imgsz: int = 640):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.imgsz = imgsz
        cuda_dnn_available = "CUDA" in cv2.getBuildInformation() and "YES" in next(
            (line for line in cv2.getBuildInformation().splitlines() if "NVIDIA CUDA" in line),
            "",
        )
        if device is not None and str(device).lower() != "cpu" and cuda_dnn_available:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except cv2.error:
                print("[WARNING] OpenCV CUDA DNN is unavailable; drowsiness model uses CPU.")
        elif device is not None and str(device).lower() != "cpu":
            print("[WARNING] OpenCV has no CUDA DNN support; drowsiness model uses CPU.")

    def predict(
        self, image: np.ndarray, conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> DrowsinessPrediction:
        predictions = self.predict_all(image, conf_threshold, iou_threshold)
        return max(
            predictions,
            key=lambda prediction: prediction.confidence,
            default=DrowsinessPrediction(0, DROWSINESS_NAMES[0], 0.0),
        )

    def predict_all(
        self, image: np.ndarray, conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[DrowsinessPrediction]:
        height, width = image.shape[:2]
        scale = min(self.imgsz / width, self.imgsz / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_w, resized_h))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - resized_w) // 2
        pad_y = (self.imgsz - resized_h) // 2
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        output_names = self.net.getUnconnectedOutLayersNames()
        decoded_name = "output" if "output" in output_names else output_names[-1]
        output = np.asarray(self.net.forward(decoded_name)).squeeze(0)
        if output.ndim != 2 or output.shape[1] < 6:
            raise RuntimeError(f"Unexpected drowsiness ONNX output shape: {output.shape}")

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in output:
            class_id = int(np.argmax(row[5:]))
            score = float(row[4] * row[5 + class_id])
            if score < conf_threshold:
                continue
            cx, cy, box_w, box_h = row[:4]
            boxes.append([
                int((cx - box_w / 2 - pad_x) / scale),
                int((cy - box_h / 2 - pad_y) / scale),
                int(box_w / scale), int(box_h / scale),
            ])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        kept = np.asarray(indices).reshape(-1).tolist() if len(indices) else []
        if not kept:
            return []
        predictions = []
        for index in kept:
            class_id = class_ids[index]
            x, y, box_w, box_h = boxes[index]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + box_w), min(height, y + box_h)
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append(DrowsinessPrediction(
                class_id,
                DROWSINESS_NAMES.get(class_id, f"class{class_id}"),
                scores[index],
                (x1, y1, x2, y2),
            ))
        return predictions
