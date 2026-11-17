import pytest

from traffic_violation_detection_vietnam.association.person_vehicle_helmet_associator import (
    AssociationStatus,
    PersonAssociation,
    PersonVehicleHelmetAssociationResult,
)
from traffic_violation_detection_vietnam.detection.helmet_detector import HelmetDetection
from traffic_violation_detection_vietnam.detection.person_vehicle_detector import ObjectDetection
from traffic_violation_detection_vietnam.violations import (
    HelmetViolationDecider,
    HelmetViolationReason,
    HelmetViolationStatus,
)


def make_association(
    motorcycle_status=AssociationStatus.ASSOCIATED,
    helmet_status=AssociationStatus.ASSOCIATED,
    helmet_class="without_helmet",
    helmet_confidence=0.9,
):
    person = ObjectDetection((10, 10, 100, 200), 0.9, 0, "person")
    motorcycle = ObjectDetection((0, 100, 150, 220), 0.9, 3, "motorcycle")

    helmet = None
    if helmet_class is not None:
        helmet = HelmetDetection(
            bbox_xyxy=(30, 15, 60, 50),
            confidence=helmet_confidence,
            class_id=1 if helmet_class == "without_helmet" else 0,
            class_name=helmet_class,
        )

    return PersonAssociation(
        person=person,
        motorcycle=(motorcycle if motorcycle_status == AssociationStatus.ASSOCIATED else None),
        helmet=helmet,
        motorcycle_status=motorcycle_status,
        helmet_status=helmet_status,
        motorcycle_score=0.8,
        helmet_score=0.8 if helmet is not None else None,
    )


def decide(association):
    result = PersonVehicleHelmetAssociationResult(
        associations=[association],
        unassigned_helmets=[],
        ambiguous_helmets=[],
        unassigned_motorcycles=[],
    )
    return HelmetViolationDecider().decide(result)[0]


def test_without_helmet_returns_violation_candidate():
    decision = decide(make_association())

    assert decision.status == HelmetViolationStatus.VIOLATION_CANDIDATE
    assert decision.reason_code == HelmetViolationReason.RIDER_WITHOUT_HELMET


def test_with_helmet_returns_no_violation_detected():
    decision = decide(make_association(helmet_class="with_helmet"))

    assert decision.status == HelmetViolationStatus.NO_VIOLATION_DETECTED


def test_low_confidence_requires_review():
    decision = decide(make_association(helmet_confidence=0.41))

    assert decision.status == HelmetViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == HelmetViolationReason.HELMET_CONFIDENCE_TOO_LOW


def test_confidence_equal_to_threshold_is_accepted():
    decision = decide(make_association(helmet_confidence=0.5))

    assert decision.status == HelmetViolationStatus.VIOLATION_CANDIDATE


def test_missing_helmet_requires_review():
    decision = decide(
        make_association(
            helmet_status=AssociationStatus.NOT_DETECTED,
            helmet_class=None,
        )
    )

    assert decision.status == HelmetViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == HelmetViolationReason.HELMET_NOT_DETECTED


def test_ambiguous_motorcycle_requires_review():
    decision = decide(make_association(motorcycle_status=AssociationStatus.AMBIGUOUS))

    assert decision.status == HelmetViolationStatus.NEEDS_REVIEW


def test_person_without_motorcycle_is_not_applicable():
    decision = decide(make_association(motorcycle_status=AssociationStatus.NOT_ASSOCIATED))

    assert decision.status == HelmetViolationStatus.NOT_APPLICABLE


def test_unknown_helmet_class_requires_review():
    decision = decide(make_association(helmet_class="unknown"))

    assert decision.status == HelmetViolationStatus.NEEDS_REVIEW
    assert decision.reason_code == HelmetViolationReason.UNKNOWN_HELMET_CLASS


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_confidence_threshold_raises_error(threshold):
    with pytest.raises(ValueError):
        HelmetViolationDecider(min_helmet_confidence=threshold)
