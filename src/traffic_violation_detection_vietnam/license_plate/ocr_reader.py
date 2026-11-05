from dataclasses import dataclass

import numpy as np
import torch  # noqa: F401
from paddleocr import PaddleOCR


@dataclass
class OCRResult:
    text: str
    confidence: float
    recognized_texts: list[str]
    recognized_scores: list[float]


class LicensePlateOCR:
    def __init__(self, device: str = "cpu") -> None:
        self.ocr = PaddleOCR(
            lang="en",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self, image: np.ndarray) -> OCRResult:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")

        results = self.ocr.predict(input=image)

        recognized_texts: list[str] = []
        recognized_scores: list[float] = []

        for result in results:
            payload = result.json
            data = payload.get("res", payload)

            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])

            for text, score in zip(texts, scores, strict=True):
                cleaned_text = str(text).strip()

                if cleaned_text:
                    recognized_texts.append(cleaned_text)
                    recognized_scores.append(float(score))

        if not recognized_texts:
            return OCRResult(
                text="",
                confidence=0.0,
                recognized_texts=[],
                recognized_scores=[],
            )

        combined_text = "".join(recognized_texts)
        average_confidence = sum(recognized_scores) / len(recognized_scores)

        return OCRResult(
            text=combined_text,
            confidence=round(average_confidence, 4),
            recognized_texts=recognized_texts,
            recognized_scores=[round(score, 4) for score in recognized_scores],
        )
