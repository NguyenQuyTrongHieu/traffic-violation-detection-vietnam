from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np


class TrafficLightState(StrEnum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TrafficLightClassification:
    state: TrafficLightState
    confidence: float
    red_ratio: float
    yellow_ratio: float
    green_ratio: float


class TrafficLightStateClassifier:
    def __init__(
        self,
        min_color_pixels: int = 3,
        min_color_ratio: float = 0.002,
        min_dominance: float = 0.60,
        min_brightness: int = 100,
        min_saturation: int = 100,
    ) -> None:
        if min_color_pixels <= 0:
            raise ValueError("min_color_pixels must be positive")

        if not 0 < min_color_ratio <= 1:
            raise ValueError("min_color_ratio must be between 0 and 1")

        if not 0 < min_dominance <= 1:
            raise ValueError("min_dominance must be between 0 and 1")

        if not 0 <= min_brightness <= 255:
            raise ValueError("min_brightness must be between 0 and 255")

        if not 0 <= min_saturation <= 255:
            raise ValueError("min_saturation must be between 0 and 255")

        self.min_color_pixels = min_color_pixels
        self.min_color_ratio = min_color_ratio
        self.min_dominance = min_dominance
        self.min_brightness = min_brightness
        self.min_saturation = min_saturation

    def predict(self, image: np.ndarray) -> TrafficLightClassification:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")

        if image.dtype != np.uint8:
            raise TypeError("Expected uint8 image")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]
        hue = hsv[:, :, 0]

        valid_pixels = (saturation >= self.min_saturation) & (brightness >= self.min_brightness)

        red_mask = valid_pixels & ((hue <= 10) | (hue >= 170))
        yellow_mask = valid_pixels & (hue >= 15) & (hue <= 35)
        green_mask = valid_pixels & (hue >= 36) & (hue <= 90)

        total_pixels = image.shape[0] * image.shape[1]

        counts = {
            TrafficLightState.RED: int(np.count_nonzero(red_mask)),
            TrafficLightState.YELLOW: int(np.count_nonzero(yellow_mask)),
            TrafficLightState.GREEN: int(np.count_nonzero(green_mask)),
        }

        ratios = {state: count / total_pixels for state, count in counts.items()}

        sorted_states = sorted(
            counts,
            key=lambda state: counts[state],
            reverse=True,
        )

        best_state = sorted_states[0]
        best_count = counts[best_state]
        second_count = counts[sorted_states[1]]

        dominance = (
            best_count / (best_count + second_count) if best_count + second_count > 0 else 0.0
        )

        if (
            best_count < self.min_color_pixels
            or ratios[best_state] < self.min_color_ratio
            or dominance < self.min_dominance
        ):
            state = TrafficLightState.UNKNOWN
            confidence = 0.0
        else:
            state = best_state
            confidence = dominance

        return TrafficLightClassification(
            state=state,
            confidence=confidence,
            red_ratio=ratios[TrafficLightState.RED],
            yellow_ratio=ratios[TrafficLightState.YELLOW],
            green_ratio=ratios[TrafficLightState.GREEN],
        )
