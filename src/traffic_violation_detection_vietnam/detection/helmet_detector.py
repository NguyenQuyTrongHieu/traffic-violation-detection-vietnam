from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

# Keep PyTorch initialization before Ultralytics on Windows.
import torch
from ultralytics import YOLO

HELMET_CLASS_NAMES = {
    0: "with_helmet",
    1: "without_helmet",
}


@dataclass
class HelmetDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class HelmetDetector:
    def __init__(
        self,
        weights_path: str | Path = "models/detection/helmet_yolo11s.pt",
        device: str | int | None = None,
        image_size: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.weights_path = Path(weights_path)

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Helmet model weights not found: {self.weights_path}")

        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = YOLO(str(self.weights_path))

        model_names = self.model.names
        for class_id, expected_name in {
            0: "With Helmet",
            1: "Without Helmet",
        }.items():
            if model_names.get(class_id) != expected_name:
                raise ValueError(
                    "Unexpected helmet model classes: "
                    f"expected class {class_id} to be "
                    f"{expected_name!r}, got {model_names.get(class_id)!r}"
                )

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[list[HelmetDetection], float]:
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
            classes=list(HELMET_CLASS_NAMES),
            verbose=False,
        )

        inference_time_ms = (perf_counter() - start_time) * 1000

        detections: list[HelmetDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in HELMET_CLASS_NAMES:
                    continue

                coordinates = box.xyxy[0].tolist()

                detections.append(
                    HelmetDetection(
                        bbox_xyxy=tuple(round(float(value), 2) for value in coordinates),
                        confidence=round(float(box.conf[0]), 4),
                        class_id=class_id,
                        class_name=HELMET_CLASS_NAMES[class_id],
                    )
                )

        return detections, round(inference_time_ms, 2)
