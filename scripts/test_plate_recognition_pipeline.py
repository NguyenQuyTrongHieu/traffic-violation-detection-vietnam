import sys
from pathlib import Path

import cv2
import torch  # noqa: F401

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
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: uv run python scripts/test_plate_recognition_pipeline.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    detector = LicensePlateDetector()
    ocr_reader = LicensePlateOCR(device="cpu")
    cropper = LicensePlateCropper(padding_ratio=0.05)

    pipeline = LicensePlateRecognitionPipeline(
        detector=detector,
        ocr_reader=ocr_reader,
        cropper=cropper,
    )

    result = pipeline.process(image)

    output_dir = Path("runs/plate_recognition")
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / f"{image_path.stem}_recognized.jpg"

    if not cv2.imwrite(
        str(annotated_path),
        result.annotated_image,
    ):
        raise RuntimeError(f"Failed to save image: {annotated_path}")

    print(f"Image size: {result.image_width}x{result.image_height}")
    print(f"Detected plates: {result.detection_count}")
    print(f"Detection time: {result.detection_time_ms:.2f} ms")
    print(f"Total processing time: {result.total_processing_time_ms:.2f} ms")

    for index, recognition in enumerate(result.recognitions):
        crop_path = crops_dir / f"plate_{index}.jpg"

        if not cv2.imwrite(str(crop_path), recognition.crop):
            raise RuntimeError(f"Failed to save crop: {crop_path}")

        print(f"\nPlate {index}:")
        print(f"  Detection confidence: {recognition.detection.confidence:.4f}")
        print(f"  OCR text: {recognition.ocr_result.text!r}")
        print(f"  OCR confidence: {recognition.ocr_result.confidence:.4f}")
        print(f"  Crop saved: {crop_path}")

    print(f"\nAnnotated image saved: {annotated_path}")


if __name__ == "__main__":
    main()
