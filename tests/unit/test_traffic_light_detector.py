from pathlib import Path

import numpy as np
import pytest

from traffic_violation_detection_vietnam.detection.traffic_light_detector import (
    TRAFFIC_LIGHT_CLASS_ID,
    TrafficLightDetector,
)


class FakeBox:
    def __init__(self, cls: int, confidence: float, xyxy: list[float]) -> None:
        self.cls = np.array([cls])
        self.conf = np.array([confidence])
        self.xyxy = np.array([xyxy])


class FakeResult:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes


class FakeYOLO:
    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(kwargs)
        return [
            FakeResult(
                [
                    FakeBox(9, 0.92, [10, 20, 50, 80]),
                    FakeBox(0, 0.88, [60, 20, 100, 100]),
                    FakeBox(3, 0.85, [110, 20, 180, 120]),
                ]
            )
        ]


def test_detector_returns_only_traffic_light_detections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "yolo11n.pt"
    weights.touch()

    fake_model = FakeYOLO(str(weights))

    monkeypatch.setattr(
        "traffic_violation_detection_vietnam.detection.traffic_light_detector.YOLO",
        lambda path: fake_model,
    )

    detector = TrafficLightDetector(weights_path=str(weights), device="cpu")
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    detections, inference_time = detector.predict(image)

    assert len(detections) == 1
    assert detections[0].class_id == TRAFFIC_LIGHT_CLASS_ID
    assert detections[0].class_name == "traffic_light"
    assert detections[0].confidence == pytest.approx(0.92)
    assert detections[0].bbox_xyxy == pytest.approx((10, 20, 50, 80))
    assert inference_time >= 0
    assert fake_model.calls[0]["classes"] == [TRAFFIC_LIGHT_CLASS_ID]


def test_detector_rejects_empty_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "yolo11n.pt"
    weights.touch()

    monkeypatch.setattr(
        "traffic_violation_detection_vietnam.detection.traffic_light_detector.YOLO",
        lambda path: FakeYOLO(path),
    )

    detector = TrafficLightDetector(weights_path=str(weights), device="cpu")

    with pytest.raises(ValueError):
        detector.predict(np.empty((0, 0, 3), dtype=np.uint8))


def test_detector_rejects_grayscale_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "yolo11n.pt"
    weights.touch()

    monkeypatch.setattr(
        "traffic_violation_detection_vietnam.detection.traffic_light_detector.YOLO",
        lambda path: FakeYOLO(path),
    )

    detector = TrafficLightDetector(weights_path=str(weights), device="cpu")

    with pytest.raises(ValueError):
        detector.predict(np.zeros((100, 100), dtype=np.uint8))


def test_detector_raises_when_weights_do_not_exist(tmp_path: Path) -> None:
    missing_weights = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        TrafficLightDetector(weights_path=str(missing_weights), device="cpu")
