from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from traffic_violation_detection_vietnam.license_plate.detector import (
    PlateDetection,
)


class DetectorProtocol(Protocol):
    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[list[PlateDetection], float]: ...


@dataclass
class LicensePlatePipelineResult:
    detections: list[PlateDetection]
    detection_count: int
    inference_time_ms: float
    image_width: int
    image_height: int
    annotated_image: np.ndarray


class LicensePlatePipeline:
    def __init__(self, detector: DetectorProtocol) -> None:
        self.detector = detector

    def process(self, image: np.ndarray) -> LicensePlatePipelineResult:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty NumPy array")

        detections, inference_time_ms = self.detector.predict(image)

        annotated_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"Plate {detection.confidence:.2f}"

            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        height, width = image.shape[:2]

        return LicensePlatePipelineResult(
            detections=detections,
            detection_count=len(detections),
            inference_time_ms=inference_time_ms,
            image_width=width,
            image_height=height,
            annotated_image=annotated_image,
        )
