import pytest

from traffic_violation_detection_vietnam.detection.signal_head_associator import (
    SignalGroupAssociation,
    SignalHeadAssociator,
    SignalHeadObservation,
)
from traffic_violation_detection_vietnam.detection.traffic_light_classifier import (
    TrafficLightState,
)


@pytest.fixture
def associator() -> SignalHeadAssociator:
    return SignalHeadAssociator(
        {
            "head_1": "group_forward",
            "head_2": "group_forward",
            "head_3": "group_forward",
            "head_4": "group_forward",
            "head_5": "group_left",
        }
    )


def test_associates_four_heads_into_one_group(
    associator: SignalHeadAssociator,
) -> None:
    observations = [SignalHeadObservation(f"head_{i}", TrafficLightState.RED) for i in range(1, 5)]

    results = associator.associate(observations)

    forward = results[0]

    assert isinstance(forward, SignalGroupAssociation)
    assert forward.group_id == "group_forward"
    assert forward.observed_head_count == 4
    assert forward.expected_head_count == 4
    assert forward.missing_head_ids == ()


def test_separates_different_signal_groups(
    associator: SignalHeadAssociator,
) -> None:
    observations = [
        SignalHeadObservation("head_1", TrafficLightState.RED),
        SignalHeadObservation("head_5", TrafficLightState.GREEN),
    ]

    results = associator.associate(observations)

    assert len(results) == 2
    assert results[0].group_id == "group_forward"
    assert results[1].group_id == "group_left"


def test_reports_missing_heads(
    associator: SignalHeadAssociator,
) -> None:
    observations = [
        SignalHeadObservation("head_1", TrafficLightState.RED),
        SignalHeadObservation("head_2", TrafficLightState.RED),
    ]

    results = associator.associate(observations)
    forward = results[0]

    assert forward.observed_head_count == 2
    assert forward.expected_head_count == 4
    assert set(forward.missing_head_ids) == {"head_3", "head_4"}


def test_preserves_inconsistent_head_states(
    associator: SignalHeadAssociator,
) -> None:
    observations = [
        SignalHeadObservation("head_1", TrafficLightState.RED),
        SignalHeadObservation("head_2", TrafficLightState.GREEN),
        SignalHeadObservation("head_3", TrafficLightState.RED),
        SignalHeadObservation("head_4", TrafficLightState.RED),
    ]

    results = associator.associate(observations)

    states = [item.state for item in results[0].head_observations]

    assert states == [
        TrafficLightState.RED,
        TrafficLightState.GREEN,
        TrafficLightState.RED,
        TrafficLightState.RED,
    ]


def test_empty_observations_return_all_heads_as_missing(
    associator: SignalHeadAssociator,
) -> None:
    results = associator.associate([])

    assert results[0].observed_head_count == 0
    assert results[0].expected_head_count == 4
    assert len(results[0].missing_head_ids) == 4


def test_unknown_head_raises_error(
    associator: SignalHeadAssociator,
) -> None:
    observations = [SignalHeadObservation("unknown_head", TrafficLightState.RED)]

    with pytest.raises(ValueError, match="Unknown signal head"):
        associator.associate(observations)


def test_duplicate_head_raises_error(
    associator: SignalHeadAssociator,
) -> None:
    observations = [
        SignalHeadObservation("head_1", TrafficLightState.RED),
        SignalHeadObservation("head_1", TrafficLightState.GREEN),
    ]

    with pytest.raises(ValueError, match="Duplicate signal head"):
        associator.associate(observations)


def test_invalid_state_raises_error(
    associator: SignalHeadAssociator,
) -> None:
    observations = [
        SignalHeadObservation("head_1", "RED"),
    ]

    with pytest.raises(TypeError, match="Invalid traffic light state"):
        associator.associate(observations)


def test_invalid_configuration_raises_error() -> None:
    with pytest.raises(ValueError):
        SignalHeadAssociator({})

    with pytest.raises(ValueError):
        SignalHeadAssociator({"head_1": ""})

    with pytest.raises(TypeError):
        SignalHeadAssociator(None)
