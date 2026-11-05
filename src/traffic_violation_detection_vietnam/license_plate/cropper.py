import math

import numpy as np


class LicensePlateCropper:
    def __init__(self, padding_ratio: float = 0.05) -> None:
        if padding_ratio < 0:
            raise ValueError("padding_ratio must be non-negative")

        self.padding_ratio = padding_ratio

    def crop(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty NumPy array")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")

        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly 4 coordinates")

        x1, y1, x2, y2 = bbox_xyxy

        if not all(math.isfinite(value) for value in bbox_xyxy):
            raise ValueError("Bounding box coordinates must be finite")

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box dimensions")

        height, width = image.shape[:2]

        padding_x = (x2 - x1) * self.padding_ratio
        padding_y = (y2 - y1) * self.padding_ratio

        x1 = max(0, math.floor(x1 - padding_x))
        y1 = max(0, math.floor(y1 - padding_y))
        x2 = min(width, math.ceil(x2 + padding_x))
        y2 = min(height, math.ceil(y2 + padding_y))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box is outside the image boundaries")

        return image[y1:y2, x1:x2].copy()
