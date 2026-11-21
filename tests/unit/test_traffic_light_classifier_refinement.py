import numpy as np
import pytest

from traffic_violation_detection_vietnam.detection.traffic_light_classifier import (
    TrafficLightState,
    TrafficLightStateClassifier,
)


@pytest.fixture
def classifier() -> TrafficLightStateClassifier:
    return TrafficLightStateClassifier()


def make_image(color: tuple[int, int, int]) -> np.ndarray:
    return np.full((50, 50, 3), color, dtype=np.uint8)


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ((0, 0, 255), TrafficLightState.RED),
        ((0, 255, 255), TrafficLightState.YELLOW),
        ((0, 255, 0), TrafficLightState.GREEN),
    ],
)
def test_detects_basic_signal_colors(
    classifier: TrafficLightStateClassifier,
    color: tuple[int, int, int],
    expected: TrafficLightState,
) -> None:
    result = classifier.predict(make_image(color))

    assert result.state == expected
    assert result.confidence >= 0.60


def test_dark_color_is_unknown(
    classifier: TrafficLightStateClassifier,
) -> None:
    result = classifier.predict(make_image((0, 0, 80)))

    assert result.state == TrafficLightState.UNKNOWN


def test_low_saturation_color_is_unknown(
    classifier: TrafficLightStateClassifier,
) -> None:
    result = classifier.predict(make_image((120, 120, 120)))

    assert result.state == TrafficLightState.UNKNOWN


def test_equal_red_and_green_is_unknown(
    classifier: TrafficLightStateClassifier,
) -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:, :25] = (0, 0, 255)
    image[:, 25:] = (0, 255, 0)

    result = classifier.predict(image)

    assert result.state == TrafficLightState.UNKNOWN


def test_small_color_region_is_unknown() -> None:
    classifier = TrafficLightStateClassifier(
        min_color_pixels=10,
        min_color_ratio=0.01,
    )

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[20:22, 20:22] = (0, 0, 255)

    result = classifier.predict(image)

    assert result.state == TrafficLightState.UNKNOWN


def test_rejects_non_uint8_image(
    classifier: TrafficLightStateClassifier,
) -> None:
    image = np.zeros((50, 50, 3), dtype=np.float32)

    with pytest.raises(TypeError):
        classifier.predict(image)
