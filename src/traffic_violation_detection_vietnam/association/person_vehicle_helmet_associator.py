from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from traffic_violation_detection_vietnam.detection.helmet_detector import (
    HelmetDetection,
)
from traffic_violation_detection_vietnam.detection.person_vehicle_detector import (
    ObjectDetection,
)


class AssociationStatus(StrEnum):
    ASSOCIATED = "associated"
    NOT_ASSOCIATED = "not_associated"
    AMBIGUOUS = "ambiguous"
    NOT_DETECTED = "not_detected"


@dataclass
class PersonAssociation:
    person: ObjectDetection
    motorcycle: ObjectDetection | None
    helmet: HelmetDetection | None
    motorcycle_status: AssociationStatus
    helmet_status: AssociationStatus
    motorcycle_score: float | None
    helmet_score: float | None


@dataclass
class PersonVehicleHelmetAssociationResult:
    associations: list[PersonAssociation]
    unassigned_helmets: list[HelmetDetection]
    ambiguous_helmets: list[HelmetDetection]
    unassigned_motorcycles: list[ObjectDetection]


class PersonVehicleHelmetAssociator:
    def __init__(
        self,
        head_region_ratio: float = 0.45,
        lower_body_start_ratio: float = 0.45,
        min_helmet_coverage: float = 0.30,
        min_vehicle_overlap: float = 0.15,
        ambiguity_margin: float = 0.10,
    ) -> None:
        ratios = {
            "head_region_ratio": head_region_ratio,
            "lower_body_start_ratio": lower_body_start_ratio,
            "min_helmet_coverage": min_helmet_coverage,
            "min_vehicle_overlap": min_vehicle_overlap,
            "ambiguity_margin": ambiguity_margin,
        }

        for name, value in ratios.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if head_region_ratio == 0:
            raise ValueError("head_region_ratio must be greater than zero")

        if lower_body_start_ratio >= 1:
            raise ValueError("lower_body_start_ratio must be less than 1")

        self.head_region_ratio = head_region_ratio
        self.lower_body_start_ratio = lower_body_start_ratio
        self.min_helmet_coverage = min_helmet_coverage
        self.min_vehicle_overlap = min_vehicle_overlap
        self.ambiguity_margin = ambiguity_margin

    @staticmethod
    def _validate_bbox(
        bbox: tuple[float, float, float, float],
    ) -> None:
        if len(bbox) != 4:
            raise ValueError("Bounding box must contain four coordinates")

        if not all(isfinite(value) for value in bbox):
            raise ValueError("Bounding box coordinates must be finite")

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box must have positive width and height")

    @staticmethod
    def _area(
        bbox: tuple[float, float, float, float],
    ) -> float:
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    @staticmethod
    def _intersection_area(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        return (x2 - x1) * (y2 - y1)

    def _head_region(
        self,
        person_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1

        return (
            x1,
            y1,
            x2,
            y1 + height * self.head_region_ratio,
        )

    def _lower_body_region(
        self,
        person_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1

        return (
            x1,
            y1 + height * self.lower_body_start_ratio,
            x2,
            y2,
        )

    def _select_candidate(
        self,
        scores: list[tuple[int, float]],
        min_score: float,
    ) -> tuple[int | None, float | None, bool]:
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)

        if not ranked or ranked[0][1] < min_score:
            return None, None, False

        if (
            len(ranked) > 1
            and ranked[1][1] >= min_score
            and ranked[0][1] - ranked[1][1] < self.ambiguity_margin
        ):
            return None, ranked[0][1], True

        return ranked[0][0], ranked[0][1], False

    def associate(
        self,
        persons: list[ObjectDetection],
        motorcycles: list[ObjectDetection],
        helmets: list[HelmetDetection],
    ) -> PersonVehicleHelmetAssociationResult:
        for detection in [*persons, *motorcycles, *helmets]:
            self._validate_bbox(detection.bbox_xyxy)

        # Associate each person with a motorcycle.
        vehicle_results: list[tuple[ObjectDetection | None, AssociationStatus, float | None]] = []

        assigned_motorcycle_indices: set[int] = set()

        for person in persons:
            lower_region = self._lower_body_region(person.bbox_xyxy)
            lower_area = self._area(lower_region)

            scores = [
                (
                    index,
                    self._intersection_area(
                        lower_region,
                        motorcycle.bbox_xyxy,
                    )
                    / lower_area,
                )
                for index, motorcycle in enumerate(motorcycles)
            ]

            index, score, ambiguous = self._select_candidate(
                scores,
                self.min_vehicle_overlap,
            )

            if ambiguous:
                vehicle_results.append((None, AssociationStatus.AMBIGUOUS, score))
            elif index is None:
                vehicle_results.append((None, AssociationStatus.NOT_ASSOCIATED, None))
            else:
                vehicle_results.append(
                    (
                        motorcycles[index],
                        AssociationStatus.ASSOCIATED,
                        score,
                    )
                )
                assigned_motorcycle_indices.add(index)

        # Associate each helmet with at most one person.
        helmet_candidates: dict[int, list[tuple[HelmetDetection, float]]] = {
            index: [] for index in range(len(persons))
        }

        ambiguous_person_indices: set[int] = set()
        ambiguous_helmet_indices: set[int] = set()

        for helmet_index, helmet in enumerate(helmets):
            helmet_area = self._area(helmet.bbox_xyxy)

            scores = []

            for person_index, person in enumerate(persons):
                head_region = self._head_region(person.bbox_xyxy)

                coverage = (
                    self._intersection_area(
                        head_region,
                        helmet.bbox_xyxy,
                    )
                    / helmet_area
                )

                scores.append((person_index, coverage))

            person_index, score, ambiguous = self._select_candidate(
                scores,
                self.min_helmet_coverage,
            )

            if ambiguous:
                ambiguous_helmet_indices.add(helmet_index)

                ranked = sorted(
                    scores,
                    key=lambda item: item[1],
                    reverse=True,
                )

                for candidate_index, candidate_score in ranked:
                    if (
                        candidate_score >= self.min_helmet_coverage
                        and score is not None
                        and score - candidate_score < self.ambiguity_margin
                    ):
                        ambiguous_person_indices.add(candidate_index)

            elif person_index is not None and score is not None:
                helmet_candidates[person_index].append((helmet, score))

        associations = []
        assigned_helmet_ids: set[int] = set()

        for person_index, person in enumerate(persons):
            motorcycle, motorcycle_status, motorcycle_score = vehicle_results[person_index]

            candidates = sorted(
                helmet_candidates[person_index],
                key=lambda item: item[1],
                reverse=True,
            )

            if candidates:
                helmet, helmet_score = candidates[0]
                assigned_helmet_ids.add(id(helmet))
                helmet_status = AssociationStatus.ASSOCIATED
            else:
                helmet = None
                helmet_score = None

                if person_index in ambiguous_person_indices:
                    helmet_status = AssociationStatus.AMBIGUOUS
                else:
                    helmet_status = AssociationStatus.NOT_DETECTED

            associations.append(
                PersonAssociation(
                    person=person,
                    motorcycle=motorcycle,
                    helmet=helmet,
                    motorcycle_status=motorcycle_status,
                    helmet_status=helmet_status,
                    motorcycle_score=(
                        round(motorcycle_score, 4) if motorcycle_score is not None else None
                    ),
                    helmet_score=(round(helmet_score, 4) if helmet_score is not None else None),
                )
            )

        unassigned_motorcycles = [
            motorcycle
            for index, motorcycle in enumerate(motorcycles)
            if index not in assigned_motorcycle_indices
        ]

        unassigned_helmets = [helmet for helmet in helmets if id(helmet) not in assigned_helmet_ids]

        ambiguous_helmets = [
            helmet for index, helmet in enumerate(helmets) if index in ambiguous_helmet_indices
        ]

        return PersonVehicleHelmetAssociationResult(
            associations=associations,
            unassigned_helmets=unassigned_helmets,
            ambiguous_helmets=ambiguous_helmets,
            unassigned_motorcycles=unassigned_motorcycles,
        )
