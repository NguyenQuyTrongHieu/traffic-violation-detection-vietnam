from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traffic_violation_detection_vietnam.detection import (
    person_vehicle_detector as detector_module,
)


class FakeModel:
    def predict(self, **kwargs):
        self.predict_kwargs = kwargs

        boxes = [
            SimpleNamespace(
                cls=[torch.tensor(0)],
                conf=[torch.tensor(0.95)],
                xyxy=[torch.tensor([10.0, 20.0, 100.0, 200.0])],
            ),
            SimpleNamespace(
                cls=[torch.tensor(3)],
                conf=[torch.tensor(0.88)],
                xyxy=[torch.tensor([120.0, 130.0, 300.0, 350.0])],
            ),
        ]

        return [SimpleNamespace(boxes=boxes)]


def create_detector(monkeypatch, tmp_path):
    fake_model = FakeModel()

    monkeypatch.setattr(
        detector_module,
        "YOLO",
        lambda weights_path: fake_model,
    )

    detector = detector_module.PersonVehicleDetector(
        weights_path=tmp_path / "models" / "yolo11n.pt",
        device="cpu",
    )

    return detector, fake_model


def test_detector_returns_person_and_motorcycle(monkeypatch, tmp_path):
    detector, fake_model = create_detector(monkeypatch, tmp_path)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections, inference_time_ms = detector.predict(image)

    assert len(detections) == 2
    assert detections[0].class_name == "person"
    assert detections[0].class_id == 0
    assert detections[1].class_name == "motorcycle"
    assert detections[1].class_id == 3
    assert inference_time_ms >= 0
    assert fake_model.predict_kwargs["classes"] == [0, 3]


def test_detector_rejects_empty_image(monkeypatch, tmp_path):
    detector, _ = create_detector(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="empty"):
        detector.predict(np.array([]))


def test_detector_rejects_invalid_image_shape(monkeypatch, tmp_path):
    detector, _ = create_detector(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="3-channel BGR"):
        detector.predict(np.zeros((100, 100), dtype=np.uint8))


def test_detector_rejects_invalid_confidence_threshold():
    with pytest.raises(ValueError, match="confidence_threshold"):
        detector_module.PersonVehicleDetector(
            confidence_threshold=1.5,
        )


def test_detector_rejects_invalid_image_size():
    with pytest.raises(ValueError, match="image_size"):
        detector_module.PersonVehicleDetector(
            image_size=0,
        )
