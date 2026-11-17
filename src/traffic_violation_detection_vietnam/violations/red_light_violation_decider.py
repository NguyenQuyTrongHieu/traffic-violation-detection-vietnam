from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class TrafficLightState(StrEnum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class CrossingStatus(StrEnum):
    CROSSED = "crossed"
    NOT_CROSSED = "not_crossed"
    AMBIGUOUS = "ambiguous"
    NOT_EVALUABLE = "not_evaluable"


class RedLightViolationStatus(StrEnum):
    VIOLATION_CANDIDATE = "violation_candidate"
    NO_VIOLATION_DETECTED = "no_violation_detected"
    NEEDS_REVIEW = "needs_review"
    NOT_APPLICABLE = "not_applicable"


class RedLightViolationReason(StrEnum):
    MOTORCYCLE_CROSSED_ON_RED = "motorcycle_crossed_on_red"
    MOTORCYCLE_CROSSED_ON_GREEN = "motorcycle_crossed_on_green"
    MOTORCYCLE_DID_NOT_CROSS = "motorcycle_did_not_cross"
    NON_MOTORCYCLE = "non_motorcycle"
    CROSSING_AMBIGUOUS = "crossing_ambiguous"
    CROSSING_NOT_EVALUABLE = "crossing_not_evaluable"
    INVALID_CROSSING_EVIDENCE = "invalid_crossing_evidence"
    TRACK_ID_MISSING = "track_id_missing"
    TRAFFIC_LIGHT_OBSERVATION_MISSING = "traffic_light_observation_missing"
    TRAFFIC_LIGHT_TIME_GAP_TOO_LARGE = "traffic_light_time_gap_too_large"
    TRAFFIC_LIGHT_STATE_TRANSITION = "traffic_light_state_transition"
    TRAFFIC_LIGHT_STATE_UNKNOWN = "traffic_light_state_unknown"
    TRAFFIC_LIGHT_YELLOW = "traffic_light_yellow"
    TRAFFIC_LIGHT_CONFIDENCE_TOO_LOW = "traffic_light_confidence_too_low"
    CROSSING_CONFIDENCE_TOO_LOW = "crossing_confidence_too_low"
    INVALID_TRAFFIC_LIGHT_OBSERVATION = "invalid_traffic_light_observation"


@dataclass
class StopLineCrossingEvent:
    track_id: str
    vehicle_class: str
    timestamp_ms: float
    status: CrossingStatus
    confidence: float


@dataclass
class TrafficLightObservation:
    timestamp_ms: float
    state: TrafficLightState
    confidence: float


@dataclass
class RedLightEvidence:
    crossing_event: StopLineCrossingEvent
    traffic_light_observations: list[TrafficLightObservation]


@dataclass
class RedLightViolationDecision:
    track_id: str
    status: RedLightViolationStatus
    reason_code: RedLightViolationReason
    crossing_timestamp_ms: float
    traffic_light_state: TrafficLightState | None
    traffic_light_confidence: float | None


class RedLightViolationDecider:
    def __init__(
        self,
        min_crossing_confidence: float = 0.5,
        min_traffic_light_confidence: float = 0.5,
        max_signal_gap_ms: float = 500.0,
    ) -> None:
        thresholds = {
            "min_crossing_confidence": min_crossing_confidence,
            "min_traffic_light_confidence": min_traffic_light_confidence,
        }

        for name, value in thresholds.items():
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if not isfinite(max_signal_gap_ms) or max_signal_gap_ms <= 0:
            raise ValueError("max_signal_gap_ms must be greater than zero")

        self.min_crossing_confidence = min_crossing_confidence
        self.min_traffic_light_confidence = min_traffic_light_confidence
        self.max_signal_gap_ms = max_signal_gap_ms

    @staticmethod
    def _valid_confidence(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and 0.0 <= value <= 1.0
        )

    def _decision(
        self,
        event: StopLineCrossingEvent,
        status: RedLightViolationStatus,
        reason: RedLightViolationReason,
        light_state: TrafficLightState | None = None,
        light_confidence: float | None = None,
    ) -> RedLightViolationDecision:
        return RedLightViolationDecision(
            track_id=event.track_id,
            status=status,
            reason_code=reason,
            crossing_timestamp_ms=event.timestamp_ms,
            traffic_light_state=light_state,
            traffic_light_confidence=light_confidence,
        )

    def _review(
        self,
        event: StopLineCrossingEvent,
        reason: RedLightViolationReason,
        light_state: TrafficLightState | None = None,
        light_confidence: float | None = None,
    ) -> RedLightViolationDecision:
        return self._decision(
            event,
            RedLightViolationStatus.NEEDS_REVIEW,
            reason,
            light_state,
            light_confidence,
        )

    def _resolve_light_state(
        self,
        event: StopLineCrossingEvent,
        observations: list[TrafficLightObservation],
    ) -> tuple[TrafficLightState | None, float | None, RedLightViolationReason | None]:
        timestamp = event.timestamp_ms
        if not isinstance(observations, list) or not observations:
            return (
                None,
                None,
                RedLightViolationReason.TRAFFIC_LIGHT_OBSERVATION_MISSING,
            )

        if not all(self._valid_observation(item) for item in observations):
            return (
                None,
                None,
                RedLightViolationReason.INVALID_TRAFFIC_LIGHT_OBSERVATION,
            )
        exact = [item for item in observations if item.timestamp_ms == timestamp]

        if exact:
            states = {item.state for item in exact}
            if len(states) != 1:
                return None, None, RedLightViolationReason.TRAFFIC_LIGHT_STATE_TRANSITION

            selected = exact
            state = exact[0].state
            confidence = min(item.confidence for item in selected)
        else:
            before = [item for item in observations if item.timestamp_ms < timestamp]
            after = [item for item in observations if item.timestamp_ms > timestamp]

            if not before or not after:
                return (
                    None,
                    None,
                    RedLightViolationReason.TRAFFIC_LIGHT_OBSERVATION_MISSING,
                )

            previous = max(before, key=lambda item: item.timestamp_ms)
            following = min(after, key=lambda item: item.timestamp_ms)

            before_gap = timestamp - previous.timestamp_ms
            after_gap = following.timestamp_ms - timestamp

            if before_gap > self.max_signal_gap_ms or after_gap > self.max_signal_gap_ms:
                return (
                    None,
                    None,
                    RedLightViolationReason.TRAFFIC_LIGHT_TIME_GAP_TOO_LARGE,
                )

            if previous.state != following.state:
                return (
                    None,
                    None,
                    RedLightViolationReason.TRAFFIC_LIGHT_STATE_TRANSITION,
                )

            state = previous.state
            confidence = min(previous.confidence, following.confidence)
            selected = [previous, following]

        if any(not self._valid_confidence(item.confidence) for item in selected):
            return (
                state,
                confidence,
                RedLightViolationReason.TRAFFIC_LIGHT_CONFIDENCE_TOO_LOW,
            )

        if confidence < self.min_traffic_light_confidence:
            return (
                state,
                confidence,
                RedLightViolationReason.TRAFFIC_LIGHT_CONFIDENCE_TOO_LOW,
            )

        if state == TrafficLightState.UNKNOWN:
            return (
                state,
                confidence,
                RedLightViolationReason.TRAFFIC_LIGHT_STATE_UNKNOWN,
            )

        return state, confidence, None

    def decide(
        self,
        evidence: RedLightEvidence,
    ) -> RedLightViolationDecision:
        event = evidence.crossing_event
        if not isinstance(event, StopLineCrossingEvent):
            raise TypeError("crossing_event must be a StopLineCrossingEvent")

        if not isinstance(event.vehicle_class, str):
            return self._review(
                event,
                RedLightViolationReason.INVALID_CROSSING_EVIDENCE,
            )

        if not isinstance(event.status, CrossingStatus):
            return self._review(
                event,
                RedLightViolationReason.INVALID_CROSSING_EVIDENCE,
            )

        if not self._valid_timestamp(event.timestamp_ms) or not self._valid_confidence(
            event.confidence
        ):
            return self._review(
                event,
                RedLightViolationReason.INVALID_CROSSING_EVIDENCE,
            )
        if event.vehicle_class != "motorcycle":
            return self._decision(
                event,
                RedLightViolationStatus.NOT_APPLICABLE,
                RedLightViolationReason.NON_MOTORCYCLE,
            )

        if event.status == CrossingStatus.NOT_CROSSED:
            return self._decision(
                event,
                RedLightViolationStatus.NO_VIOLATION_DETECTED,
                RedLightViolationReason.MOTORCYCLE_DID_NOT_CROSS,
            )

        if event.status == CrossingStatus.AMBIGUOUS:
            return self._review(
                event,
                RedLightViolationReason.CROSSING_AMBIGUOUS,
            )

        if event.status == CrossingStatus.NOT_EVALUABLE:
            return self._review(
                event,
                RedLightViolationReason.CROSSING_NOT_EVALUABLE,
            )

        if event.status != CrossingStatus.CROSSED:
            return self._review(
                event,
                RedLightViolationReason.INVALID_CROSSING_EVIDENCE,
            )

        if not isinstance(event.track_id, str):
            return self._review(
                event,
                RedLightViolationReason.INVALID_CROSSING_EVIDENCE,
            )

        if not event.track_id.strip():
            return self._review(
                event,
                RedLightViolationReason.TRACK_ID_MISSING,
            )

        if event.confidence < self.min_crossing_confidence:
            return self._review(
                event,
                RedLightViolationReason.CROSSING_CONFIDENCE_TOO_LOW,
            )

        state, confidence, error = self._resolve_light_state(
            event,
            evidence.traffic_light_observations,
        )

        if error is not None:
            return self._review(event, error, state, confidence)

        if state == TrafficLightState.RED:
            return self._decision(
                event,
                RedLightViolationStatus.VIOLATION_CANDIDATE,
                RedLightViolationReason.MOTORCYCLE_CROSSED_ON_RED,
                state,
                confidence,
            )

        if state == TrafficLightState.GREEN:
            return self._decision(
                event,
                RedLightViolationStatus.NO_VIOLATION_DETECTED,
                RedLightViolationReason.MOTORCYCLE_CROSSED_ON_GREEN,
                state,
                confidence,
            )

        if state == TrafficLightState.YELLOW:
            return self._review(
                event,
                RedLightViolationReason.TRAFFIC_LIGHT_YELLOW,
                state,
                confidence,
            )

        return self._review(
            event,
            RedLightViolationReason.TRAFFIC_LIGHT_STATE_UNKNOWN,
            state,
            confidence,
        )

    @staticmethod
    def _valid_timestamp(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value >= 0
        )

    @classmethod
    def _valid_observation(
        cls,
        observation: TrafficLightObservation,
    ) -> bool:
        return (
            isinstance(observation, TrafficLightObservation)
            and cls._valid_timestamp(observation.timestamp_ms)
            and isinstance(observation.state, TrafficLightState)
            and cls._valid_confidence(observation.confidence)
        )
