import sys
from pathlib import Path

import cv2
import torch

from traffic_violation_detection_vietnam.detection.person_vehicle_detector import (
    PersonVehicleDetector,
)

CLASS_COLORS = {
    "person": (255, 0, 0),
    "motorcycle": (0, 255, 0),
}


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: uv run python scripts/test_person_vehicle_detector.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    detector = PersonVehicleDetector()
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
            0.6,
            color,
            2,
        )

    output_dir = Path("runs/person_vehicle_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_detected.jpg"

    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError(f"Cannot save output image: {output_path}")

    person_count = sum(detection.class_name == "person" for detection in detections)
    motorcycle_count = sum(detection.class_name == "motorcycle" for detection in detections)

    print("Image size:", image.shape[1], "x", image.shape[0])
    print("Detection count:", len(detections))
    print("Person count:", person_count)
    print("Motorcycle count:", motorcycle_count)
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
