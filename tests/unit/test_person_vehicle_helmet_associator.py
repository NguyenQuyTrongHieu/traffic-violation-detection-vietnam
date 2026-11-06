import pytest

from traffic_violation_detection_vietnam.association import (
    AssociationStatus,
    PersonVehicleHelmetAssociator,
)
from traffic_violation_detection_vietnam.detection.helmet_detector import (
    HelmetDetection,
)
from traffic_violation_detection_vietnam.detection.person_vehicle_detector import (
    ObjectDetection,
)


def make_person(bbox):
    return ObjectDetection(
        bbox_xyxy=bbox,
        confidence=0.95,
        class_id=0,
        class_name="person",
    )


def make_motorcycle(bbox):
    return ObjectDetection(
        bbox_xyxy=bbox,
        confidence=0.90,
        class_id=3,
        class_name="motorcycle",
    )


def make_helmet(bbox):
    return HelmetDetection(
        bbox_xyxy=bbox,
        confidence=0.92,
        class_id=0,
        class_name="with_helmet",
    )


def test_associates_person_helmet_and_motorcycle():
    associator = PersonVehicleHelmetAssociator()

    person = make_person((100, 50, 200, 250))
    motorcycle = make_motorcycle((80, 150, 230, 280))
    helmet = make_helmet((130, 60, 170, 100))

    result = associator.associate(
        persons=[person],
        motorcycles=[motorcycle],
        helmets=[helmet],
    )

    association = result.associations[0]

    assert association.motorcycle is motorcycle
    assert association.helmet is helmet
    assert association.motorcycle_status == AssociationStatus.ASSOCIATED
    assert association.helmet_status == AssociationStatus.ASSOCIATED


def test_multiple_people_can_share_one_motorcycle():
    associator = PersonVehicleHelmetAssociator()

    persons = [
        make_person((100, 50, 150, 250)),
        make_person((160, 50, 210, 250)),
    ]
    motorcycle = make_motorcycle((80, 150, 240, 280))

    result = associator.associate(
        persons=persons,
        motorcycles=[motorcycle],
        helmets=[],
    )

    assert len(result.associations) == 2
    assert all(item.motorcycle is motorcycle for item in result.associations)


def test_missing_helmet_is_not_detected_not_violation():
    associator = PersonVehicleHelmetAssociator()

    person = make_person((100, 50, 200, 250))

    result = associator.associate(
        persons=[person],
        motorcycles=[],
        helmets=[],
    )

    association = result.associations[0]

    assert association.helmet is None
    assert association.helmet_status == AssociationStatus.NOT_DETECTED


def test_ambiguous_helmet_is_not_assigned():
    associator = PersonVehicleHelmetAssociator()

    persons = [
        make_person((100, 50, 200, 250)),
        make_person((140, 50, 240, 250)),
    ]
    helmet = make_helmet((145, 60, 175, 100))

    result = associator.associate(
        persons=persons,
        motorcycles=[],
        helmets=[helmet],
    )

    assert helmet in result.ambiguous_helmets
    assert all(item.helmet is None for item in result.associations)
    assert all(item.helmet_status == AssociationStatus.AMBIGUOUS for item in result.associations)


def test_ambiguous_motorcycle_is_not_assigned():
    associator = PersonVehicleHelmetAssociator()

    person = make_person((100, 50, 200, 250))
    motorcycles = [
        make_motorcycle((90, 140, 160, 260)),
        make_motorcycle((140, 140, 210, 260)),
    ]

    result = associator.associate(
        persons=[person],
        motorcycles=motorcycles,
        helmets=[],
    )

    association = result.associations[0]

    assert association.motorcycle is None
    assert association.motorcycle_status == AssociationStatus.AMBIGUOUS


def test_rejects_invalid_bounding_box():
    associator = PersonVehicleHelmetAssociator()

    person = make_person((100, 100, 50, 200))

    with pytest.raises(ValueError, match="positive width"):
        associator.associate(
            persons=[person],
            motorcycles=[],
            helmets=[],
        )


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="min_helmet_coverage"):
        PersonVehicleHelmetAssociator(min_helmet_coverage=1.5)
