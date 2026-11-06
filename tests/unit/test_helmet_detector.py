from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traffic_violation_detection_vietnam.detection import helmet_detector


class FakeModel:
    names = {
        0: "With Helmet",
        1: "Without Helmet",
    }

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs

        boxes = [
            SimpleNamespace(
                cls=[torch.tensor(0)],
                conf=[torch.tensor(0.95)],
                xyxy=[torch.tensor([10.0, 20.0, 100.0, 120.0])],
            ),
            SimpleNamespace(
                cls=[torch.tensor(1)],
                conf=[torch.tensor(0.88)],
                xyxy=[torch.tensor([120.0, 30.0, 210.0, 130.0])],
            ),
        ]

        return [SimpleNamespace(boxes=boxes)]


def create_detector(monkeypatch, tmp_path):
    fake_model = FakeModel()
    weights_path = tmp_path / "helmet_yolo11s.pt"
    weights_path.touch()

    monkeypatch.setattr(
        helmet_detector,
        "YOLO",
        lambda weights_path: fake_model,
    )

    detector = helmet_detector.HelmetDetector(
        weights_path=weights_path,
        device="cpu",
    )

    return detector, fake_model


def test_detector_returns_both_helmet_classes(monkeypatch, tmp_path):
    detector, fake_model = create_detector(monkeypatch, tmp_path)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections, inference_time_ms = detector.predict(image)

    assert len(detections) == 2
    assert detections[0].class_name == "with_helmet"
    assert detections[0].class_id == 0
    assert detections[1].class_name == "without_helmet"
    assert detections[1].class_id == 1
    assert inference_time_ms >= 0
    assert fake_model.predict_kwargs["classes"] == [0, 1]


def test_detector_rejects_empty_image(monkeypatch, tmp_path):
    detector, _ = create_detector(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="empty"):
        detector.predict(np.array([]))


def test_detector_rejects_invalid_image_shape(monkeypatch, tmp_path):
    detector, _ = create_detector(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="3-channel BGR"):
        detector.predict(np.zeros((100, 100), dtype=np.uint8))


def test_detector_rejects_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights not found"):
        helmet_detector.HelmetDetector(
            weights_path=tmp_path / "missing.pt",
        )


def test_detector_rejects_invalid_confidence_threshold():
    with pytest.raises(ValueError, match="confidence_threshold"):
        helmet_detector.HelmetDetector(
            confidence_threshold=1.5,
        )


def test_detector_rejects_invalid_image_size():
    with pytest.raises(ValueError, match="image_size"):
        helmet_detector.HelmetDetector(
            image_size=0,
        )
