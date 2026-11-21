from dataclasses import dataclass

from traffic_violation_detection_vietnam.detection.traffic_light_classifier import (
    TrafficLightState,
)


@dataclass(frozen=True)
class SignalHeadObservation:
    head_id: str
    state: TrafficLightState


@dataclass(frozen=True)
class SignalGroupAssociation:
    group_id: str
    head_observations: tuple[SignalHeadObservation, ...]
    missing_head_ids: tuple[str, ...]

    @property
    def observed_head_count(self) -> int:
        return len(self.head_observations)

    @property
    def expected_head_count(self) -> int:
        return self.observed_head_count + len(self.missing_head_ids)


class SignalHeadAssociator:
    def __init__(self, head_to_group: dict[str, str]) -> None:
        if not isinstance(head_to_group, dict):
            raise TypeError("head_to_group must be a dictionary")

        if not head_to_group:
            raise ValueError("head_to_group cannot be empty")

        for head_id, group_id in head_to_group.items():
            if not isinstance(head_id, str) or not head_id.strip():
                raise ValueError("Every head_id must be a non-empty string")

            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError("Every group_id must be a non-empty string")

        self._head_to_group = head_to_group.copy()

        self._group_to_heads: dict[str, tuple[str, ...]] = {}

        for head_id, group_id in self._head_to_group.items():
            self._group_to_heads.setdefault(group_id, tuple())
            self._group_to_heads[group_id] += (head_id,)

    def associate(
        self,
        observations: list[SignalHeadObservation],
    ) -> tuple[SignalGroupAssociation, ...]:
        if not isinstance(observations, list):
            raise TypeError("observations must be a list")

        observed_by_id: dict[str, SignalHeadObservation] = {}

        for observation in observations:
            if not isinstance(observation, SignalHeadObservation):
                raise TypeError("Every observation must be a SignalHeadObservation")

            head_id = observation.head_id

            if not isinstance(head_id, str) or not head_id.strip():
                raise ValueError("head_id must be a non-empty string")

            if head_id not in self._head_to_group:
                raise ValueError(f"Unknown signal head: {head_id}")

            if head_id in observed_by_id:
                raise ValueError(f"Duplicate signal head: {head_id}")

            if not isinstance(observation.state, TrafficLightState):
                raise TypeError("Invalid traffic light state")

            observed_by_id[head_id] = observation

        results = []

        for group_id, expected_head_ids in self._group_to_heads.items():
            group_observations = tuple(
                observed_by_id[head_id]
                for head_id in expected_head_ids
                if head_id in observed_by_id
            )

            missing_head_ids = tuple(
                head_id for head_id in expected_head_ids if head_id not in observed_by_id
            )

            results.append(
                SignalGroupAssociation(
                    group_id=group_id,
                    head_observations=group_observations,
                    missing_head_ids=missing_head_ids,
                )
            )

        return tuple(results)
