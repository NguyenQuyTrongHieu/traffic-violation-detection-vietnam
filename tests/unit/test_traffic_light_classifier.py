import numpy as np
import pytest

from traffic_violation_detection_vietnam.detection.traffic_light_classifier import (
    TrafficLightState,
    TrafficLightStateClassifier,
)


@pytest.fixture
def classifier() -> TrafficLightStateClassifier:
    return TrafficLightStateClassifier()


def make_color_image(
    bgr: tuple[int, int, int],
    height: int = 50,
    width: int = 50,
) -> np.ndarray:
    return np.full((height, width, 3), bgr, dtype=np.uint8)


def test_classifies_red_light(
    classifier: TrafficLightStateClassifier,
) -> None:
    result = classifier.predict(make_color_image((0, 0, 255)))

    assert result.state == TrafficLightState.RED
    assert result.red_ratio > 0
    assert result.confidence >= 0.60


def test_classifies_yellow_light(
    classifier: TrafficLightStateClassifier,
) -> None:
    result = classifier.predict(make_color_image((0, 255, 255)))

    assert result.state == TrafficLightState.YELLOW
    assert result.yellow_ratio > 0
    assert result.confidence >= 0.60


def test_classifies_green_light(
    classifier: TrafficLightStateClassifier,
) -> None:
    result = classifier.predict(make_color_image((0, 255, 0)))

    assert result.state == TrafficLightState.GREEN
    assert result.green_ratio > 0
    assert result.confidence >= 0.60


def test_classifies_black_image_as_unknown(
    classifier: TrafficLightStateClassifier,
) -> None:
    result = classifier.predict(make_color_image((0, 0, 0)))

    assert result.state == TrafficLightState.UNKNOWN


def test_classifies_ambiguous_red_green_image_as_unknown(
    classifier: TrafficLightStateClassifier,
) -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:, :25] = (0, 0, 255)
    image[:, 25:] = (0, 255, 0)

    result = classifier.predict(image)

    assert result.state == TrafficLightState.UNKNOWN


def test_rejects_empty_image(
    classifier: TrafficLightStateClassifier,
) -> None:
    with pytest.raises(ValueError):
        classifier.predict(np.empty((0, 0, 3), dtype=np.uint8))


def test_rejects_grayscale_image(
    classifier: TrafficLightStateClassifier,
) -> None:
    with pytest.raises(ValueError):
        classifier.predict(np.zeros((50, 50), dtype=np.uint8))
