from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from traffic_violation_detection_vietnam.detection.traffic_light_classifier import (
    TrafficLightState,
)


class SignalGroupState(StrEnum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass(frozen=True)
class SignalGroupObservation:
    group_id: str
    head_states: tuple[TrafficLightState, ...]
    group_state: SignalGroupState
    observed_at_ms: float
    expected_head_count: int
    observed_head_count: int
    is_consistent: bool


class SignalGroupAggregator:
    def aggregate(
        self,
        group_id: str,
        head_states: list[TrafficLightState],
        expected_head_count: int,
        observed_at_ms: float,
    ) -> SignalGroupObservation:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group_id must be a non-empty string")

        if not isinstance(head_states, list):
            raise TypeError("head_states must be a list")

        if not isinstance(expected_head_count, int) or isinstance(expected_head_count, bool):
            raise TypeError("expected_head_count must be an integer")

        if expected_head_count <= 0:
            raise ValueError("expected_head_count must be positive")

        if not isinstance(observed_at_ms, (int, float)) or isinstance(observed_at_ms, bool):
            raise TypeError("observed_at_ms must be numeric")

        if not isfinite(observed_at_ms) or observed_at_ms < 0:
            raise ValueError("observed_at_ms must be finite and non-negative")

        if not all(isinstance(state, TrafficLightState) for state in head_states):
            raise TypeError("Every head state must be a TrafficLightState")

        observed_count = len(head_states)
        unique_states = set(head_states)

        group_state = SignalGroupState.NEEDS_REVIEW
        is_consistent = False

        if (
            observed_count == expected_head_count
            and observed_count > 0
            and len(unique_states) == 1
            and TrafficLightState.UNKNOWN not in unique_states
        ):
            state = head_states[0]
            group_state = SignalGroupState(state.value)
            is_consistent = True

        return SignalGroupObservation(
            group_id=group_id,
            head_states=tuple(head_states),
            group_state=group_state,
            observed_at_ms=float(observed_at_ms),
            expected_head_count=expected_head_count,
            observed_head_count=observed_count,
            is_consistent=is_consistent,
        )
