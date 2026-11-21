import pytest

from traffic_violation_detection_vietnam.detection.signal_group_aggregator import (
    SignalGroupAggregator,
    SignalGroupState,
)
from traffic_violation_detection_vietnam.detection.traffic_light_classifier import (
    TrafficLightState,
)


@pytest.fixture
def aggregator() -> SignalGroupAggregator:
    return SignalGroupAggregator()


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (TrafficLightState.RED, SignalGroupState.RED),
        (TrafficLightState.GREEN, SignalGroupState.GREEN),
        (TrafficLightState.YELLOW, SignalGroupState.YELLOW),
    ],
)
def test_all_heads_same_state(
    aggregator: SignalGroupAggregator,
    state: TrafficLightState,
    expected: SignalGroupState,
) -> None:
    result = aggregator.aggregate(
        group_id="group_1",
        head_states=[state, state, state, state],
        expected_head_count=4,
        observed_at_ms=1000,
    )

    assert result.group_state == expected
    assert result.is_consistent is True
    assert result.observed_head_count == 4


@pytest.mark.parametrize(
    "states",
    [
        [
            TrafficLightState.RED,
            TrafficLightState.RED,
            TrafficLightState.RED,
            TrafficLightState.GREEN,
        ],
        [
            TrafficLightState.RED,
            TrafficLightState.RED,
            TrafficLightState.YELLOW,
            TrafficLightState.YELLOW,
        ],
    ],
)
def test_inconsistent_heads_need_review(
    aggregator: SignalGroupAggregator,
    states: list[TrafficLightState],
) -> None:
    result = aggregator.aggregate("group_1", states, 4, 1000)

    assert result.group_state == SignalGroupState.NEEDS_REVIEW
    assert result.is_consistent is False


def test_unknown_head_requires_review(
    aggregator: SignalGroupAggregator,
) -> None:
    states = [
        TrafficLightState.RED,
        TrafficLightState.RED,
        TrafficLightState.RED,
        TrafficLightState.UNKNOWN,
    ]

    result = aggregator.aggregate("group_1", states, 4, 1000)

    assert result.group_state == SignalGroupState.NEEDS_REVIEW
    assert result.is_consistent is False


def test_missing_head_requires_review(
    aggregator: SignalGroupAggregator,
) -> None:
    states = [
        TrafficLightState.RED,
        TrafficLightState.RED,
        TrafficLightState.RED,
    ]

    result = aggregator.aggregate("group_1", states, 4, 1000)

    assert result.group_state == SignalGroupState.NEEDS_REVIEW
    assert result.observed_head_count == 3


def test_empty_head_states_require_review(
    aggregator: SignalGroupAggregator,
) -> None:
    result = aggregator.aggregate("group_1", [], 4, 1000)

    assert result.group_state == SignalGroupState.NEEDS_REVIEW
    assert result.is_consistent is False


def test_invalid_group_id_raises_error(
    aggregator: SignalGroupAggregator,
) -> None:
    with pytest.raises(ValueError):
        aggregator.aggregate("", [TrafficLightState.RED], 1, 1000)


def test_invalid_expected_head_count_raises_error(
    aggregator: SignalGroupAggregator,
) -> None:
    with pytest.raises(ValueError):
        aggregator.aggregate("group_1", [TrafficLightState.RED], 0, 1000)


def test_invalid_timestamp_raises_error(
    aggregator: SignalGroupAggregator,
) -> None:
    with pytest.raises(ValueError):
        aggregator.aggregate("group_1", [TrafficLightState.RED], 1, -1)


def test_invalid_state_type_raises_error(
    aggregator: SignalGroupAggregator,
) -> None:
    with pytest.raises(TypeError):
        aggregator.aggregate("group_1", ["RED"], 1, 1000)
