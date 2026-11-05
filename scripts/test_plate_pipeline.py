import sys
from pathlib import Path

import cv2
import torch

from traffic_violation_detection_vietnam.license_plate.detector import (
    LicensePlateDetector,
)
from traffic_violation_detection_vietnam.license_plate.pipeline import (
    LicensePlatePipeline,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: python scripts/test_plate_pipeline.py <image>")

    image_path = Path(sys.argv[1])
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector = LicensePlateDetector(device="0")
    pipeline = LicensePlatePipeline(detector)

    result = pipeline.process(image)

    output_dir = Path("runs/plate_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), result.annotated_image)

    print("CUDA available:", torch.cuda.is_available())
    print("Image:", result.image_width, "x", result.image_height)
    print("Detected plates:", result.detection_count)
    print("Inference time:", result.inference_time_ms, "ms")
    print("Output:", output_path.resolve())

    for detection in result.detections:
        print(detection)


if __name__ == "__main__":
    main()
