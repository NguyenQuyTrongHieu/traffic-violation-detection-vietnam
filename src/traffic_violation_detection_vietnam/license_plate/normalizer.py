import re


def normalize_plate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Plate text must be a string")

    normalized = text.upper().strip()
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)

    return normalized


def compare_plate_text(detected_text: str, expected_text: str) -> bool:
    detected = normalize_plate_text(detected_text)
    expected = normalize_plate_text(expected_text)

    if not detected or not expected:
        return False

    return detected == expected
