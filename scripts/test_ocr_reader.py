import sys
from pathlib import Path

import cv2

from traffic_violation_detection_vietnam.license_plate.ocr_reader import (
    LicensePlateOCR,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: python scripts/test_ocr_reader.py <image_path>")

    image_path = Path(sys.argv[1])
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    ocr = LicensePlateOCR(device="cpu")
    result = ocr.predict(image)

    print("Recognized text:", result.text)
    print("OCR confidence:", result.confidence)
    print("Recognized segments:", result.recognized_texts)
    print("Segment scores:", result.recognized_scores)


if __name__ == "__main__":
    main()
