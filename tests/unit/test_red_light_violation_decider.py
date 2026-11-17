import pytest

from traffic_violation_detection_vietnam.violations.red_light_violation_decider import (
    CrossingStatus,
    RedLightEvidence,
    RedLightViolationDecider,
    RedLightViolationReason,
    RedLightViolationStatus,
    StopLineCrossingEvent,
    TrafficLightObservation,
    TrafficLightState,
)


def make_evidence(
    crossing_status=CrossingStatus.CROSSED,
    vehicle_class="motorcycle",
    crossing_confidence=0.9,
    light_before=TrafficLightState.RED,
    light_after=TrafficLightState.RED,
    light_confidence=0.9,
    before_timestamp=900.0,
    crossing_timestamp=1000.0,
    after_timestamp=1100.0,
):
    event = StopLineCrossingEvent(
        track_id="track-001",
        vehicle_class=vehicle_class,
        timestamp_ms=crossing_timestamp,
        status=crossing_status,
        confidence=crossing_confidence,
    )

    observations = [
        TrafficLightObservation(
            timestamp_ms=before_timestamp,
            state=light_before,
            confidence=light_confidence,
        ),
        TrafficLightObservation(
            timestamp_ms=after_timestamp,
            state=light_after,
            confidence=light_confidence,
        ),
    ]

    return RedLightEvidence(
        crossing_event=event,
        traffic_light_observations=observations,
    )


def test_motorcycle_crossing_on_red_is_violation_candidate():
    decision = RedLightViolationDecider().decide(make_evidence())

    assert decision.status == RedLightViolationStatus.VIOLATION_CANDIDATE
    assert decision.reason_code == RedLightViolationReason.MOTORCYCLE_CROSSED_ON_RED
    assert decision.traffic_light_state == TrafficLightState.RED


def test_motorcycle_crossing_on_green_is_no_violation():
    evidence = make_evidence(
        light_before=TrafficLightState.GREEN,
        light_after=TrafficLightState.GREEN,
    )

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NO_VIOLATION_DETECTED


def test_yellow_light_requires_review():
    evidence = make_evidence(
        light_before=TrafficLightState.YELLOW,
        light_after=TrafficLightState.YELLOW,
    )

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == RedLightViolationReason.TRAFFIC_LIGHT_YELLOW


def test_light_transition_during_crossing_requires_review():
    evidence = make_evidence(
        light_before=TrafficLightState.RED,
        light_after=TrafficLightState.GREEN,
    )

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == RedLightViolationReason.TRAFFIC_LIGHT_STATE_TRANSITION


def test_missing_light_observation_requires_review():
    evidence = make_evidence()
    evidence.traffic_light_observations = [evidence.traffic_light_observations[0]]

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == RedLightViolationReason.TRAFFIC_LIGHT_OBSERVATION_MISSING


def test_large_light_observation_gap_requires_review():
    evidence = make_evidence(
        before_timestamp=100.0,
        after_timestamp=2000.0,
        crossing_timestamp=1000.0,
    )

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == RedLightViolationReason.TRAFFIC_LIGHT_TIME_GAP_TOO_LARGE


def test_low_light_confidence_requires_review():
    evidence = make_evidence(light_confidence=0.3)

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == RedLightViolationReason.TRAFFIC_LIGHT_CONFIDENCE_TOO_LOW


def test_low_crossing_confidence_requires_review():
    evidence = make_evidence(crossing_confidence=0.3)

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == RedLightViolationReason.CROSSING_CONFIDENCE_TOO_LOW


def test_not_crossed_is_no_violation():
    evidence = make_evidence(crossing_status=CrossingStatus.NOT_CROSSED)

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NO_VIOLATION_DETECTED


def test_ambiguous_crossing_requires_review():
    evidence = make_evidence(crossing_status=CrossingStatus.AMBIGUOUS)

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW


def test_not_evaluable_crossing_requires_review():
    evidence = make_evidence(crossing_status=CrossingStatus.NOT_EVALUABLE)

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW


def test_non_motorcycle_is_not_applicable():
    evidence = make_evidence(vehicle_class="car")

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NOT_APPLICABLE


def test_unknown_traffic_light_state_requires_review():
    evidence = make_evidence(
        light_before=TrafficLightState.UNKNOWN,
        light_after=TrafficLightState.UNKNOWN,
    )

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == RedLightViolationReason.TRAFFIC_LIGHT_STATE_UNKNOWN


def test_exact_timestamp_light_observation_can_be_used():
    evidence = make_evidence()
    evidence.traffic_light_observations = [
        TrafficLightObservation(
            timestamp_ms=1000.0,
            state=TrafficLightState.RED,
            confidence=0.9,
        )
    ]

    decision = RedLightViolationDecider().decide(evidence)

    assert decision.status == RedLightViolationStatus.VIOLATION_CANDIDATE


@pytest.mark.parametrize(
    ("crossing_threshold", "light_threshold", "max_gap"),
    [
        (-0.1, 0.5, 500.0),
        (0.5, 1.1, 500.0),
        (0.5, 0.5, 0.0),
        (float("nan"), 0.5, 500.0),
    ],
)
def test_invalid_configuration_raises_error(
    crossing_threshold,
    light_threshold,
    max_gap,
):
    with pytest.raises(ValueError):
        RedLightViolationDecider(
            min_crossing_confidence=crossing_threshold,
            min_traffic_light_confidence=light_threshold,
            max_signal_gap_ms=max_gap,
        )
