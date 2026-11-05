from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

import cv2
import numpy as np

from traffic_violation_detection_vietnam.license_plate.cropper import (
    LicensePlateCropper,
)
from traffic_violation_detection_vietnam.license_plate.detector import (
    PlateDetection,
)
from traffic_violation_detection_vietnam.license_plate.normalizer import (
    normalize_plate_text,
)
from traffic_violation_detection_vietnam.license_plate.ocr_reader import (
    LicensePlateOCR,
    OCRResult,
)


class PlateMatchStatus(StrEnum):
    NOT_CHECKED = "NOT_CHECKED"
    PLATE_MATCH = "PLATE_MATCH"
    PLATE_MISMATCH = "PLATE_MISMATCH"
    PLATE_UNREADABLE = "PLATE_UNREADABLE"


@dataclass
class PlateRecognition:
    detection: PlateDetection
    ocr_result: OCRResult
    crop: np.ndarray
    normalized_text: str
    match_status: PlateMatchStatus

    @property
    def is_match(self) -> bool | None:
        if self.match_status == PlateMatchStatus.NOT_CHECKED:
            return None

        return self.match_status == PlateMatchStatus.PLATE_MATCH


@dataclass
class LicensePlateRecognitionResult:
    recognitions: list[PlateRecognition]
    detection_count: int
    detection_time_ms: float
    total_processing_time_ms: float
    image_width: int
    image_height: int
    annotated_image: np.ndarray
    expected_plate: str | None
    status: PlateMatchStatus


class LicensePlateRecognitionPipeline:
    def __init__(
        self,
        detector,
        ocr_reader: LicensePlateOCR,
        cropper: LicensePlateCropper | None = None,
    ) -> None:
        self.detector = detector
        self.ocr_reader = ocr_reader
        self.cropper = cropper or LicensePlateCropper()

    def process(
        self,
        image: np.ndarray,
        expected_plate: str | None = None,
    ) -> LicensePlateRecognitionResult:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty NumPy array")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")

        normalized_expected = None

        if expected_plate is not None:
            normalized_expected = normalize_plate_text(expected_plate)

            if not normalized_expected:
                raise ValueError("Expected plate must contain letters or digits")

        start_time = perf_counter()

        detections, detection_time_ms = self.detector.predict(image)

        annotated_image = image.copy()
        recognitions: list[PlateRecognition] = []

        for detection in detections:
            crop = self.cropper.crop(image, detection.bbox_xyxy)
            ocr_result = self.ocr_reader.predict(crop)

            normalized_text = normalize_plate_text(ocr_result.text)

            if normalized_expected is None:
                match_status = PlateMatchStatus.NOT_CHECKED
            elif not normalized_text:
                match_status = PlateMatchStatus.PLATE_UNREADABLE
            elif normalized_text == normalized_expected:
                match_status = PlateMatchStatus.PLATE_MATCH
            else:
                match_status = PlateMatchStatus.PLATE_MISMATCH

            recognition = PlateRecognition(
                detection=detection,
                ocr_result=ocr_result,
                crop=crop,
                normalized_text=normalized_text,
                match_status=match_status,
            )
            recognitions.append(recognition)

            x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)

            if match_status == PlateMatchStatus.PLATE_MATCH:
                color = (0, 255, 0)
                match_label = "MATCH"
            elif match_status == PlateMatchStatus.PLATE_MISMATCH:
                color = (0, 0, 255)
                match_label = "MISMATCH"
            elif match_status == PlateMatchStatus.PLATE_UNREADABLE:
                color = (0, 165, 255)
                match_label = "UNREADABLE"
            else:
                color = (0, 255, 0)
                match_label = ""

            recognized_text = normalized_text or "UNREADABLE"

            if match_label:
                label = f"{recognized_text} | {match_label} ({detection.confidence:.2f})"
            else:
                label = f"{recognized_text} ({detection.confidence:.2f})"

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        if normalized_expected is None:
            overall_status = PlateMatchStatus.NOT_CHECKED
        elif any(item.match_status == PlateMatchStatus.PLATE_MATCH for item in recognitions):
            overall_status = PlateMatchStatus.PLATE_MATCH
        elif not recognitions or all(
            item.match_status == PlateMatchStatus.PLATE_UNREADABLE for item in recognitions
        ):
            overall_status = PlateMatchStatus.PLATE_UNREADABLE
        else:
            overall_status = PlateMatchStatus.PLATE_MISMATCH

        height, width = image.shape[:2]
        total_processing_time_ms = (perf_counter() - start_time) * 1000

        return LicensePlateRecognitionResult(
            recognitions=recognitions,
            detection_count=len(recognitions),
            detection_time_ms=round(detection_time_ms, 2),
            total_processing_time_ms=round(total_processing_time_ms, 2),
            image_width=width,
            image_height=height,
            annotated_image=annotated_image,
            expected_plate=normalized_expected,
            status=overall_status,
        )
