import pytest

from traffic_violation_detection_vietnam.detection.signal_group_aggregator import (
    SignalGroupObservation,
    SignalGroupState,
)
from traffic_violation_detection_vietnam.detection.temporal_signal_state_validator import (
    TemporalSignalStateValidator,
)


@pytest.fixture
def validator() -> TemporalSignalStateValidator:
    return TemporalSignalStateValidator(
        min_consecutive_observations=3,
        min_stable_duration_ms=500,
        max_observation_gap_ms=1000,
    )


def make_observation(
    timestamp: float,
    state: SignalGroupState = SignalGroupState.RED,
    *,
    consistent: bool = True,
) -> SignalGroupObservation:
    return SignalGroupObservation(
        group_id="group_1",
        head_states=(),
        group_state=state,
        observed_at_ms=timestamp,
        expected_head_count=4,
        observed_head_count=4 if consistent else 3,
        is_consistent=consistent,
    )


def test_red_becomes_stable_after_three_observations_and_500ms(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))
    validator.update(make_observation(250))
    result = validator.update(make_observation(500))

    assert result.state == SignalGroupState.RED
    assert result.is_stable is True
    assert result.consecutive_observations == 3
    assert result.candidate_duration_ms == 500


def test_green_becomes_stable_after_three_observations_and_500ms(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0, SignalGroupState.GREEN))
    validator.update(make_observation(250, SignalGroupState.GREEN))
    result = validator.update(make_observation(500, SignalGroupState.GREEN))

    assert result.state == SignalGroupState.GREEN
    assert result.is_stable is True


def test_state_change_from_stable_red_returns_transition(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))
    validator.update(make_observation(250))
    validator.update(make_observation(500))

    result = validator.update(make_observation(750, SignalGroupState.GREEN))

    assert result.state == SignalGroupState.TRANSITION
    assert result.is_stable is False


def test_new_state_becomes_stable_after_transition(
    validator: TemporalSignalStateValidator,
) -> None:
    for timestamp in (0, 250, 500):
        validator.update(make_observation(timestamp))

    validator.update(make_observation(750, SignalGroupState.GREEN))
    validator.update(make_observation(1000, SignalGroupState.GREEN))
    result = validator.update(make_observation(1250, SignalGroupState.GREEN))

    assert result.state == SignalGroupState.GREEN
    assert result.is_stable is True


def test_alternating_states_never_become_stable(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))
    validator.update(make_observation(250, SignalGroupState.GREEN))
    result = validator.update(make_observation(500))

    assert result.state == SignalGroupState.NEEDS_REVIEW
    assert result.is_stable is False


def test_inconsistent_signal_group_requires_review(
    validator: TemporalSignalStateValidator,
) -> None:
    result = validator.update(make_observation(0, consistent=False))

    assert result.state == SignalGroupState.NEEDS_REVIEW
    assert result.is_stable is False
    assert result.consecutive_observations == 0


def test_unknown_state_requires_review_and_resets_history(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))

    result = validator.update(make_observation(250, SignalGroupState.UNKNOWN))

    assert result.state == SignalGroupState.NEEDS_REVIEW
    assert result.consecutive_observations == 0


def test_observation_gap_resets_stability_candidate(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))
    validator.update(make_observation(250))

    result = validator.update(make_observation(1500))

    assert result.state == SignalGroupState.NEEDS_REVIEW
    assert result.consecutive_observations == 1
    assert result.is_stable is False


@pytest.mark.parametrize("timestamp", [100, 0])
def test_non_increasing_timestamp_raises_error(
    validator: TemporalSignalStateValidator,
    timestamp: float,
) -> None:
    validator.update(make_observation(100))

    with pytest.raises(ValueError, match="timestamps must increase"):
        validator.update(make_observation(timestamp))


def test_signal_groups_have_independent_histories(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))
    validator.update(make_observation(250))
    validator.update(make_observation(500))

    group_2 = SignalGroupObservation(
        group_id="group_2",
        head_states=(),
        group_state=SignalGroupState.GREEN,
        observed_at_ms=100,
        expected_head_count=2,
        observed_head_count=2,
        is_consistent=True,
    )

    result = validator.update(group_2)

    assert result.group_id == "group_2"
    assert result.state == SignalGroupState.NEEDS_REVIEW
    assert result.consecutive_observations == 1


def test_insufficient_observation_count_does_not_become_stable(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))
    result = validator.update(make_observation(500))

    assert result.state == SignalGroupState.NEEDS_REVIEW
    assert result.consecutive_observations == 2
    assert result.is_stable is False


def test_insufficient_duration_does_not_become_stable(
    validator: TemporalSignalStateValidator,
) -> None:
    validator.update(make_observation(0))
    validator.update(make_observation(100))
    result = validator.update(make_observation(200))

    assert result.state == SignalGroupState.NEEDS_REVIEW
    assert result.consecutive_observations == 3
    assert result.candidate_duration_ms == 200
    assert result.is_stable is False


def test_invalid_observation_type_raises_error(
    validator: TemporalSignalStateValidator,
) -> None:
    with pytest.raises(TypeError):
        validator.update("RED")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_value",
    [0, -1, float("nan"), float("inf"), True],
)
def test_invalid_stable_duration_raises_error(
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError):
        TemporalSignalStateValidator(
            min_stable_duration_ms=invalid_value,
        )


def test_invalid_observation_count_raises_error() -> None:
    with pytest.raises(ValueError):
        TemporalSignalStateValidator(min_consecutive_observations=0)


def test_invalid_observation_gap_raises_error() -> None:
    with pytest.raises(ValueError):
        TemporalSignalStateValidator(max_observation_gap_ms=0)


def test_return_to_stable_state_resets_transition_candidate(validator):
    validator.update(make_observation(0, SignalGroupState.RED))
    validator.update(make_observation(250, SignalGroupState.RED))
    validator.update(make_observation(500, SignalGroupState.RED))

    # Bắt đầu chuyển RED -> GREEN
    validator.update(make_observation(750, SignalGroupState.GREEN))

    # Quay lại RED: phải hủy candidate GREEN cũ
    result = validator.update(make_observation(1000, SignalGroupState.RED))
    assert result.state == SignalGroupState.RED

    # GREEN bắt đầu lại, phải tính từ đầu
    validator.update(make_observation(1250, SignalGroupState.GREEN))
    validator.update(make_observation(1500, SignalGroupState.GREEN))

    result = validator.update(make_observation(1750, SignalGroupState.GREEN))
    assert result.state == SignalGroupState.GREEN
    assert result.is_stable is True
