import sys
from pathlib import Path

import cv2
import torch

from traffic_violation_detection_vietnam.license_plate.cropper import (
    LicensePlateCropper,
)
from traffic_violation_detection_vietnam.license_plate.detector import (
    LicensePlateDetector,
)
from traffic_violation_detection_vietnam.license_plate.ocr_reader import (
    LicensePlateOCR,
)
from traffic_violation_detection_vietnam.license_plate.recognition_pipeline import (
    LicensePlateRecognitionPipeline,
)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: uv run python scripts/test_plate_matching.py <image_path> <expected_plate>"
        )

    image_path = Path(sys.argv[1])
    expected_plate = sys.argv[2]

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    device = "0" if torch.cuda.is_available() else "cpu"

    detector = LicensePlateDetector(device=device)
    cropper = LicensePlateCropper(padding_ratio=0.05)
    ocr = LicensePlateOCR(device="cpu")

    pipeline = LicensePlateRecognitionPipeline(
        detector=detector,
        ocr_reader=ocr,
        cropper=cropper,
    )

    result = pipeline.process(
        image=image,
        expected_plate=expected_plate,
    )

    output_dir = Path("runs/plate_matching")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_matched.jpg"

    if not cv2.imwrite(str(output_path), result.annotated_image):
        raise RuntimeError(f"Failed to save annotated image: {output_path}")

    print("\n========== PLATE MATCHING RESULT ==========")
    print(f"Expected plate: {result.expected_plate}")
    print(f"Overall status: {result.status.value}")
    print(f"Detected plates: {result.detection_count}")
    print(f"Detection time: {result.detection_time_ms:.2f} ms")
    print(f"Total processing time: {result.total_processing_time_ms:.2f} ms")

    for index, recognition in enumerate(result.recognitions):
        print(f"\nPlate {index + 1}:")
        print(f"  Detection confidence: {recognition.detection.confidence:.4f}")
        print(f"  OCR text: {recognition.ocr_result.text!r}")
        print(f"  Normalized: {recognition.normalized_text!r}")
        print(f"  OCR confidence: {recognition.ocr_result.confidence:.4f}")
        print(f"  Match status: {recognition.match_status.value}")

    print(f"\nAnnotated image: {output_path}")


if __name__ == "__main__":
    main()
