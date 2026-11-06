import sys
from pathlib import Path

import cv2
import torch

from traffic_violation_detection_vietnam.detection.helmet_detector import (
    HelmetDetector,
)

CLASS_COLORS = {
    "with_helmet": (0, 255, 0),
    "without_helmet": (0, 0, 255),
}


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: uv run python scripts/test_helmet_detector.py <image_path>")

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    detector = HelmetDetector()
    detections, inference_time_ms = detector.predict(image)

    annotated_image = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
        color = CLASS_COLORS[detection.class_name]
        label = f"{detection.class_name} ({detection.confidence:.2f})"

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

    output_dir = Path("runs/helmet_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_helmet_detected.jpg"

    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError(f"Cannot save output image: {output_path}")

    with_helmet_count = sum(detection.class_name == "with_helmet" for detection in detections)
    without_helmet_count = sum(detection.class_name == "without_helmet" for detection in detections)

    print("Image size:", image.shape[1], "x", image.shape[0])
    print("Detection count:", len(detections))
    print("With helmet:", with_helmet_count)
    print("Without helmet:", without_helmet_count)
    print("Inference time:", inference_time_ms, "ms")

    for detection in detections:
        print(
            {
                "class": detection.class_name,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.bbox_xyxy,
            }
        )

    print("Output saved to:", output_path)


if __name__ == "__main__":
    main()
