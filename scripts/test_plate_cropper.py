import sys
from pathlib import Path

import cv2

from traffic_violation_detection_vietnam.license_plate.cropper import (
    LicensePlateCropper,
)
from traffic_violation_detection_vietnam.license_plate.detector import (
    LicensePlateDetector,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: uv run python scripts/test_plate_cropper.py <image_path>")

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    detector = LicensePlateDetector()
    cropper = LicensePlateCropper(padding_ratio=0.05)

    detections, inference_time_ms = detector.predict(image)

    output_dir = Path("runs/plate_crops")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected plates: {len(detections)}")
    print(f"Detection time: {inference_time_ms:.2f} ms")

    for index, detection in enumerate(detections):
        crop = cropper.crop(image, detection.bbox_xyxy)

        output_path = output_dir / f"plate_{index}.jpg"

        if not cv2.imwrite(str(output_path), crop):
            raise RuntimeError(f"Failed to save crop: {output_path}")

        height, width = crop.shape[:2]

        print(
            f"Plate {index}: "
            f"confidence={detection.confidence:.4f}, "
            f"crop_size={width}x{height}, "
            f"saved={output_path}"
        )


if __name__ == "__main__":
    main()
