from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

# Keep PyTorch initialization before Ultralytics on Windows.
import torch  # noqa: F401
from ultralytics import YOLO


@dataclass
class PlateDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


class LicensePlateDetector:
    def __init__(
        self,
        weights_path: str | Path = "models/license_plate/yolov8n_edge.pt",
        device: str = "0",
        image_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        self.device = device
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = YOLO(str(self.weights_path))

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[list[PlateDetection], float]:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")

        start_time = perf_counter()

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = (perf_counter() - start_time) * 1000

        detections: list[PlateDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()

                detections.append(
                    PlateDetection(
                        bbox_xyxy=tuple(round(float(value), 2) for value in coordinates),
                        confidence=round(float(box.conf[0]), 4),
                        class_id=int(box.cls[0]),
                    )
                )

        return detections, round(inference_time_ms, 2)
