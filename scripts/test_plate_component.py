import sys

import cv2
import torch

from traffic_violation_detection_vietnam.license_plate.detector import (
    LicensePlateDetector,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: python scripts/test_plate_component.py <image_path>")

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    detector = LicensePlateDetector()
    detections, inference_time_ms = detector.predict(image)

    print("Image size:", image.shape[1], "x", image.shape[0])
    print("Detection count:", len(detections))
    print("Inference time:", inference_time_ms, "ms")

    for detection in detections:
        print(detection)


if __name__ == "__main__":
    main()
