from dataclasses import dataclass
from math import isfinite

from traffic_violation_detection_vietnam.detection.signal_group_aggregator import (
    SignalGroupObservation,
    SignalGroupState,
)


@dataclass
class _GroupHistory:
    last_timestamp_ms: float | None = None
    stable_state: SignalGroupState | None = None
    candidate_state: SignalGroupState | None = None
    candidate_since_ms: float | None = None
    candidate_count: int = 0


@dataclass(frozen=True)
class TemporalSignalStateResult:
    group_id: str
    state: SignalGroupState
    observed_at_ms: float
    consecutive_observations: int
    candidate_duration_ms: float
    is_stable: bool


class TemporalSignalStateValidator:
    VALID_STATES = {
        SignalGroupState.RED,
        SignalGroupState.YELLOW,
        SignalGroupState.GREEN,
    }

    def __init__(
        self,
        min_consecutive_observations: int = 3,
        min_stable_duration_ms: float = 500.0,
        max_observation_gap_ms: float = 1000.0,
    ) -> None:
        if (
            not isinstance(min_consecutive_observations, int)
            or isinstance(min_consecutive_observations, bool)
            or min_consecutive_observations <= 0
        ):
            raise ValueError("min_consecutive_observations must be positive")

        for name, value in (
            ("min_stable_duration_ms", min_stable_duration_ms),
            ("max_observation_gap_ms", max_observation_gap_ms),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")

        self.min_consecutive_observations = min_consecutive_observations
        self.min_stable_duration_ms = float(min_stable_duration_ms)
        self.max_observation_gap_ms = float(max_observation_gap_ms)
        self._history: dict[str, _GroupHistory] = {}

    def update(
        self,
        observation: SignalGroupObservation,
    ) -> TemporalSignalStateResult:
        if not isinstance(observation, SignalGroupObservation):
            raise TypeError("observation must be a SignalGroupObservation")

        group_id = observation.group_id
        timestamp = observation.observed_at_ms

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("Invalid group_id")

        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("Invalid observation timestamp")

        history = self._history.setdefault(group_id, _GroupHistory())

        if history.last_timestamp_ms is not None and timestamp <= history.last_timestamp_ms:
            raise ValueError("Observation timestamps must increase")

        gap_exceeded = (
            history.last_timestamp_ms is not None
            and timestamp - history.last_timestamp_ms > self.max_observation_gap_ms
        )

        history.last_timestamp_ms = float(timestamp)

        if gap_exceeded:
            history.stable_state = None
            history.candidate_state = None
            history.candidate_since_ms = None
            history.candidate_count = 0

        if not observation.is_consistent or observation.group_state not in self.VALID_STATES:
            history.stable_state = None
            history.candidate_state = None
            history.candidate_since_ms = None
            history.candidate_count = 0

            return self._result(
                group_id,
                SignalGroupState.NEEDS_REVIEW,
                timestamp,
                history,
                False,
            )

        current_state = observation.group_state

        if current_state == history.stable_state:
            history.candidate_state = None
            history.candidate_since_ms = None
            history.candidate_count = 0

            return self._result(
                group_id,
                current_state,
                timestamp,
                history,
                True,
            )

        if current_state != history.candidate_state:
            history.candidate_state = current_state
            history.candidate_since_ms = float(timestamp)
            history.candidate_count = 1
        else:
            history.candidate_count += 1

        candidate_duration = (
            float(timestamp) - history.candidate_since_ms
            if history.candidate_since_ms is not None
            else 0.0
        )

        is_stable = (
            history.candidate_count >= self.min_consecutive_observations
            and candidate_duration >= self.min_stable_duration_ms
        )

        if is_stable:
            history.stable_state = current_state
            result_state = current_state
        elif history.stable_state is not None:
            result_state = SignalGroupState.TRANSITION
        else:
            result_state = SignalGroupState.NEEDS_REVIEW

        return self._result(
            group_id,
            result_state,
            timestamp,
            history,
            is_stable,
        )

    @staticmethod
    def _result(
        group_id: str,
        state: SignalGroupState,
        timestamp: float,
        history: _GroupHistory,
        is_stable: bool,
    ) -> TemporalSignalStateResult:
        duration = (
            timestamp - history.candidate_since_ms
            if history.candidate_since_ms is not None
            else 0.0
        )

        return TemporalSignalStateResult(
            group_id=group_id,
            state=state,
            observed_at_ms=float(timestamp),
            consecutive_observations=history.candidate_count,
            candidate_duration_ms=duration,
            is_stable=is_stable,
        )
