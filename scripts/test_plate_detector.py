import sys
from pathlib import Path

import torch

# Import PyTorch trước các thư viện inference khác.
from ultralytics import YOLO


def main() -> None:
    weights = Path("models/license_plate/yolov8n_edge.pt")

    if len(sys.argv) < 2:
        raise ValueError("Usage: python scripts/test_plate_detector.py <image_path>")

    image_path = Path(sys.argv[1])

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stop GPU inference test.")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(weights))

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.10,
        device=0,
        save=True,
        project="runs/plate_detection",
        name="smoke_test",
        exist_ok=True,
        verbose=False,
    )

    for result in results:
        print("Detected plates:", len(result.boxes))

        for box in result.boxes:
            print(
                {
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )

    print("Output saved to: runs/plate_detection/smoke_test/")


if __name__ == "__main__":
    main()
