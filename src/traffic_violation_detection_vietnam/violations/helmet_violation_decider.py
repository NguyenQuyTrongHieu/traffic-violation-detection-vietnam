from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from traffic_violation_detection_vietnam.association.person_vehicle_helmet_associator import (
    AssociationStatus,
    PersonAssociation,
    PersonVehicleHelmetAssociationResult,
)
from traffic_violation_detection_vietnam.detection.helmet_detector import HelmetDetection
from traffic_violation_detection_vietnam.detection.person_vehicle_detector import ObjectDetection


class HelmetViolationStatus(StrEnum):
    VIOLATION_CANDIDATE = "violation_candidate"
    NO_VIOLATION_DETECTED = "no_violation_detected"
    NEEDS_REVIEW = "needs_review"
    NOT_APPLICABLE = "not_applicable"


class HelmetViolationReason(StrEnum):
    RIDER_WITHOUT_HELMET = "rider_without_helmet"
    RIDER_WITH_HELMET = "rider_with_helmet"
    MOTORCYCLE_NOT_ASSOCIATED = "motorcycle_not_associated"
    MOTORCYCLE_ASSOCIATION_AMBIGUOUS = "motorcycle_association_ambiguous"
    MOTORCYCLE_DATA_MISSING = "motorcycle_data_missing"
    HELMET_NOT_DETECTED = "helmet_not_detected"
    HELMET_ASSOCIATION_AMBIGUOUS = "helmet_association_ambiguous"
    HELMET_DATA_INCONSISTENT = "helmet_data_inconsistent"
    HELMET_CONFIDENCE_INVALID = "helmet_confidence_invalid"
    HELMET_CONFIDENCE_TOO_LOW = "helmet_confidence_too_low"
    UNKNOWN_HELMET_CLASS = "unknown_helmet_class"
    MOTORCYCLE_DATA_INCONSISTENT = "motorcycle_data_inconsistent"
    HELMET_CLASS_INCONSISTENT = "helmet_class_inconsistent"


@dataclass
class HelmetViolationDecision:
    person: ObjectDetection
    motorcycle: ObjectDetection | None
    helmet: HelmetDetection | None
    status: HelmetViolationStatus
    reason_code: HelmetViolationReason
    helmet_confidence: float | None


class HelmetViolationDecider:
    def __init__(self, min_helmet_confidence: float = 0.5) -> None:
        if not isfinite(min_helmet_confidence) or not 0.0 <= min_helmet_confidence <= 1.0:
            raise ValueError("min_helmet_confidence must be between 0 and 1")

        self.min_helmet_confidence = min_helmet_confidence

    def _decision(
        self,
        association: PersonAssociation,
        status: HelmetViolationStatus,
        reason_code: HelmetViolationReason,
    ) -> HelmetViolationDecision:
        return HelmetViolationDecision(
            person=association.person,
            motorcycle=association.motorcycle,
            helmet=association.helmet,
            status=status,
            reason_code=reason_code,
            helmet_confidence=(
                association.helmet.confidence if association.helmet is not None else None
            ),
        )

    def _decide_person(
        self,
        association: PersonAssociation,
    ) -> HelmetViolationDecision:
        if association.motorcycle_status in {
            AssociationStatus.NOT_ASSOCIATED,
            AssociationStatus.NOT_DETECTED,
        }:
            if association.motorcycle is not None:
                return self._decision(
                    association,
                    HelmetViolationStatus.NEEDS_REVIEW,
                    HelmetViolationReason.MOTORCYCLE_DATA_INCONSISTENT,
                )

            return self._decision(
                association,
                HelmetViolationStatus.NOT_APPLICABLE,
                HelmetViolationReason.MOTORCYCLE_NOT_ASSOCIATED,
            )

        if association.motorcycle_status == AssociationStatus.AMBIGUOUS:
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.MOTORCYCLE_ASSOCIATION_AMBIGUOUS,
            )

        if association.motorcycle is None:
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.MOTORCYCLE_DATA_MISSING,
            )

        if association.helmet_status == AssociationStatus.AMBIGUOUS:
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.HELMET_ASSOCIATION_AMBIGUOUS,
            )

        if association.helmet_status != AssociationStatus.ASSOCIATED:
            reason = (
                HelmetViolationReason.HELMET_NOT_DETECTED
                if association.helmet is None
                else HelmetViolationReason.HELMET_DATA_INCONSISTENT
            )
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                reason,
            )

        helmet = association.helmet

        if helmet is None:
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.HELMET_DATA_INCONSISTENT,
            )
        expected_classes = {
            0: "with_helmet",
            1: "without_helmet",
        }

        if helmet.class_name not in expected_classes.values():
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.UNKNOWN_HELMET_CLASS,
            )

        expected_class_name = expected_classes.get(helmet.class_id)

        if expected_class_name is None:
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.UNKNOWN_HELMET_CLASS,
            )

        if helmet.class_name != expected_class_name:
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.HELMET_CLASS_INCONSISTENT,
            )
        if (
            not isinstance(helmet.confidence, (int, float))
            or isinstance(helmet.confidence, bool)
            or not isfinite(helmet.confidence)
            or not 0.0 <= helmet.confidence <= 1.0
        ):
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.HELMET_CONFIDENCE_INVALID,
            )

        if helmet.confidence < self.min_helmet_confidence:
            return self._decision(
                association,
                HelmetViolationStatus.NEEDS_REVIEW,
                HelmetViolationReason.HELMET_CONFIDENCE_TOO_LOW,
            )

        if helmet.class_name == "without_helmet":
            return self._decision(
                association,
                HelmetViolationStatus.VIOLATION_CANDIDATE,
                HelmetViolationReason.RIDER_WITHOUT_HELMET,
            )

        if helmet.class_name == "with_helmet":
            return self._decision(
                association,
                HelmetViolationStatus.NO_VIOLATION_DETECTED,
                HelmetViolationReason.RIDER_WITH_HELMET,
            )

        return self._decision(
            association,
            HelmetViolationStatus.NEEDS_REVIEW,
            HelmetViolationReason.UNKNOWN_HELMET_CLASS,
        )

    def decide(
        self,
        association_result: PersonVehicleHelmetAssociationResult,
    ) -> list[HelmetViolationDecision]:
        return [self._decide_person(association) for association in association_result.associations]
