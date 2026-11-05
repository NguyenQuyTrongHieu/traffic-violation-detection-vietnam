import pytest

from traffic_violation_detection_vietnam.license_plate.normalizer import (
    compare_plate_text,
    normalize_plate_text,
)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("29.H1339.79", "29H133979"),
        ("51aB - 123.45", "51AB12345"),
        ("30-B1 123.45", "30B112345"),
        ("  59A.123-45  ", "59A12345"),
        ("", ""),
    ],
)
def test_normalize_plate_text(raw_text: str, expected: str) -> None:
    assert normalize_plate_text(raw_text) == expected


def test_compare_matching_plates() -> None:
    assert compare_plate_text("29.H1339.79", "29H133979") is True


def test_compare_different_plates() -> None:
    assert compare_plate_text("29.H1339.79", "29H1339.78") is False


def test_empty_plate_does_not_match() -> None:
    assert compare_plate_text("", "") is False


def test_normalize_plate_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_plate_text(None)
