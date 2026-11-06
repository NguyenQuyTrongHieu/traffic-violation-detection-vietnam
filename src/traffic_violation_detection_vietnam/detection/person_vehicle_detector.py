from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

# Keep PyTorch initialization before Ultralytics on Windows.
import torch
from ultralytics import YOLO

COCO_CLASS_NAMES = {
    0: "person",
    3: "motorcycle",
}


@dataclass
class ObjectDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class PersonVehicleDetector:
    def __init__(
        self,
        weights_path: str | Path = "models/detection/yolo11n.pt",
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
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)

        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Ultralytics downloads yolo11n.pt automatically if it is missing.
        self.model = YOLO(str(self.weights_path))

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[list[ObjectDetection], float]:
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
            classes=list(COCO_CLASS_NAMES),
            verbose=False,
        )

        inference_time_ms = (perf_counter() - start_time) * 1000

        detections: list[ObjectDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in COCO_CLASS_NAMES:
                    continue

                coordinates = box.xyxy[0].tolist()

                detections.append(
                    ObjectDetection(
                        bbox_xyxy=tuple(round(float(value), 2) for value in coordinates),
                        confidence=round(float(box.conf[0]), 4),
                        class_id=class_id,
                        class_name=COCO_CLASS_NAMES[class_id],
                    )
                )

        return detections, round(inference_time_ms, 2)
