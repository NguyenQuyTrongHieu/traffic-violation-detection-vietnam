import argparse
from pathlib import Path

import cv2
import torch

from traffic_violation_detection_vietnam.association import (
    AssociationStatus,
    PersonVehicleHelmetAssociator,
)
from traffic_violation_detection_vietnam.detection.helmet_detector import (
    HelmetDetector,
)
from traffic_violation_detection_vietnam.detection.person_vehicle_detector import (
    PersonVehicleDetector,
)

COLORS = {
    "person": (255, 0, 0),
    "motorcycle": (0, 255, 0),
    "with_helmet": (0, 255, 255),
    "without_helmet": (0, 0, 255),
}


def draw_box(image, bbox, color, label):
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def main():
    parser = argparse.ArgumentParser(description="Smoke test person-motorcycle-helmet association.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image = cv2.imread(str(args.image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {args.image_path}")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 1. Run person and motorcycle detector.
    person_vehicle_detector = PersonVehicleDetector()
    object_detections, pv_time = person_vehicle_detector.predict(image)

    persons = [detection for detection in object_detections if detection.class_name == "person"]

    motorcycles = [
        detection for detection in object_detections if detection.class_name == "motorcycle"
    ]

    # 2. Run helmet detector.
    helmet_detector = HelmetDetector()
    helmets, helmet_time = helmet_detector.predict(image)

    # 3. Associate detections.
    associator = PersonVehicleHelmetAssociator()
    result = associator.associate(
        persons=persons,
        motorcycles=motorcycles,
        helmets=helmets,
    )

    # 4. Draw all detected objects.
    annotated_image = image.copy()

    for motorcycle in motorcycles:
        draw_box(
            annotated_image,
            motorcycle.bbox_xyxy,
            COLORS["motorcycle"],
            f"motorcycle {motorcycle.confidence:.2f}",
        )

    for helmet in helmets:
        draw_box(
            annotated_image,
            helmet.bbox_xyxy,
            COLORS[helmet.class_name],
            f"{helmet.class_name} {helmet.confidence:.2f}",
        )

    # 5. Draw associations.
    print("\nAssociation results:")

    for index, association in enumerate(result.associations):
        person = association.person

        person_label = (
            f"person | vehicle: {association.motorcycle_status.value}"
            f" | helmet: {association.helmet_status.value}"
        )

        draw_box(
            annotated_image,
            person.bbox_xyxy,
            COLORS["person"],
            person_label,
        )

        person_center = get_center(person.bbox_xyxy)

        if association.motorcycle is not None:
            motorcycle_center = get_center(association.motorcycle.bbox_xyxy)
            cv2.line(
                annotated_image,
                person_center,
                motorcycle_center,
                (255, 255, 0),
                2,
            )

        if association.helmet is not None:
            helmet_center = get_center(association.helmet.bbox_xyxy)
            cv2.line(
                annotated_image,
                person_center,
                helmet_center,
                (0, 165, 255),
                2,
            )

        print(
            {
                "person_index": index,
                "motorcycle_status": association.motorcycle_status.value,
                "motorcycle_score": association.motorcycle_score,
                "helmet_status": association.helmet_status.value,
                "helmet_score": association.helmet_score,
                "helmet_class": (
                    association.helmet.class_name if association.helmet is not None else None
                ),
            }
        )

    # 6. Save annotated image.
    output_path = args.output

    if output_path is None:
        output_dir = Path("runs/person_vehicle_helmet_association")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.image_path.stem}_associated.jpg"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), annotated_image):
        raise RuntimeError(f"Cannot save output image: {output_path}")

    # 7. Print summary.
    vehicle_associated = sum(
        item.motorcycle_status == AssociationStatus.ASSOCIATED for item in result.associations
    )

    vehicle_ambiguous = sum(
        item.motorcycle_status == AssociationStatus.AMBIGUOUS for item in result.associations
    )

    helmet_associated = sum(
        item.helmet_status == AssociationStatus.ASSOCIATED for item in result.associations
    )

    helmet_ambiguous = sum(
        item.helmet_status == AssociationStatus.AMBIGUOUS for item in result.associations
    )

    print("\n========== SUMMARY ==========")
    print("Image size:", image.shape[1], "x", image.shape[0])
    print("Person detections:", len(persons))
    print("Motorcycle detections:", len(motorcycles))
    print("Helmet detections:", len(helmets))
    print("Vehicle associations:", vehicle_associated)
    print("Ambiguous vehicle associations:", vehicle_ambiguous)
    print("Helmet associations:", helmet_associated)
    print("Ambiguous helmet associations:", helmet_ambiguous)
    print("Unassigned motorcycles:", len(result.unassigned_motorcycles))
    print("Unassigned helmets:", len(result.unassigned_helmets))
    print("Person/vehicle inference:", pv_time, "ms")
    print("Helmet inference:", helmet_time, "ms")
    print("Output:", output_path)


if __name__ == "__main__":
    main()
