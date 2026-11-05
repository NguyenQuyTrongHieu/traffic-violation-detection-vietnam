
# VTVDS — Implementation Registry

**Project:** Vietnam Traffic Violation Detection System
**Document:** Implementation Registry
**Version:** 0.1
**Current series:** T0-007 — OCR Integration
**Purpose:** Maintain a canonical reference for implemented modules, classes, functions, imports, and interfaces.

---

## 1. Source-of-truth rules

- The actual source code in the Git repository is the final source of truth.
- This registry documents the public interfaces and relationships between components.
- Before proposing code, verify the existing class names, function names, constructor arguments, and method signatures.
- Never invent or rename an existing API without explicitly explaining why.
- When an API changes, update this registry in the same ticket.
- Do not create duplicate components that perform the same responsibility.

## 2. License plate module

Package path:

`src/traffic_violation_detection_vietnam/license_plate/`

### 2.1 `detector.py`

**Class:** `PlateDetection`

Fields:

- `bbox_xyxy`: tuple of four float coordinates.
- `confidence`: detection confidence.
- `class_id`: detected class ID.

**Class:** `LicensePlateDetector`

Responsibilities:

- Load YOLOv8n license plate detection weights.
- Run inference on a BGR image.
- Return detected plate bounding boxes and inference time.

Known constructor interface:

```python
LicensePlateDetector(
    weights_path="models/license_plate/yolov8n_edge.pt",
    device="0",
    image_size=640,
    confidence_threshold=0.25,
    iou_threshold=0.45,
)
```

Known method:

```python
predict(
    image: np.ndarray,
) -> tuple[list[PlateDetection], float]
```

Returns:

- List of `PlateDetection`.
- Inference time in milliseconds.

**Windows import requirement:** Import `torch` before `ultralytics`.

### 2.2 `cropper.py`

**Class:** `LicensePlateCropper`

Responsibility:

- Crop a detected license plate from the original BGR image.
- Apply configurable padding.
- Clip crop coordinates to image boundaries.
- Reject invalid or out-of-image bounding boxes.

Known constructor:

```python
LicensePlateCropper(padding_ratio: float = 0.05)
```

Known method:

```python
crop(
    image: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> np.ndarray
```

Returns an independent cropped image.

### 2.3 `ocr_reader.py`

**Class:** `OCRResult`

Fields:

- `text: str`
- `confidence: float`
- `recognized_texts: list[str]`
- `recognized_scores: list[float]`

**Class:** `LicensePlateOCR`

Responsibility:

- Recognize text from a cropped license plate image using PaddleOCR.

Known constructor:

```python
LicensePlateOCR(device: str = "cpu")
```

Known method:

```python
predict(image: np.ndarray) -> OCRResult
```

Current baseline:

- PaddleOCR 3.4.1
- PP-OCRv5 server detection model
- English mobile recognition model
- CPU inference

**Windows import requirement:** Import `torch` before `paddleocr`.

Important:

- OCR confidence is not a calibrated probability that the complete plate is correct.
- Empty OCR output must be handled.
- Current OCR output may contain separators such as dots and hyphens.

### 2.4 `normalizer.py`

Functions:

```python
normalize_plate_text(text: str) -> str
```

Responsibilities:

- Convert text to uppercase.
- Remove non-alphanumeric characters.
- Preserve only Latin letters and digits.

```python
compare_plate_text(
    detected_text: str,
    expected_text: str,
) -> bool
```

Responsibilities:

- Normalize both inputs.
- Perform exact comparison.
- Return `False` if either normalized value is empty.
- Do not use fuzzy matching.

### 2.5 `recognition_pipeline.py`

**Confirmed class name from the user's local traceback:**

`LicensePlateRecognitionPipeline`

Do not use the alternative name `PlateRecognitionPipeline`.

Expected component dependencies:

- `LicensePlateDetector`
- `LicensePlateCropper`
- `LicensePlateOCR`
- Plate normalization and comparison functions

The implementation is intended to orchestrate:

```text
Image
  → Detection
  → Cropping
  → OCR
  → Normalization
  → Exact Matching
  → Recognition Result
```

The exact constructor and method signatures must be verified against the current local file before further integration.

Known result concepts from the implementation:

- `PLATE_MATCH`
- `PLATE_MISMATCH`
- `PLATE_UNREADABLE`

The pipeline must process every detected plate, not only the first detection.

---

## 3. Existing detector-only pipeline

### `pipeline.py`

**Class:** `DetectorProtocol`

Expected interface:

```python
predict(
    image: np.ndarray,
) -> tuple[list[PlateDetection], float]
```

**Class:** `LicensePlatePipeline`

Responsibility:

- Run plate detection.
- Draw detection bounding boxes and confidence labels.
- Return an annotated image and detection metadata.

Known result class:

`LicensePlatePipelineResult`

Fields:

- `detections`
- `detection_count`
- `inference_time_ms`
- `image_width`
- `image_height`
- `annotated_image`

This detector-only pipeline is separate from the integrated OCR recognition pipeline.

---

## 4. Current interface dependency map

```text
LicensePlateDetector
        |
        | predict(image)
        v
PlateDetection[]
        |
        v
LicensePlateCropper
        |
        | crop(image, bbox_xyxy)
        v
Cropped plate image
        |
        v
LicensePlateOCR
        |
        | predict(crop)
        v
OCRResult
        |
        v
normalize_plate_text()
        |
        v
Exact plate comparison
        |
        v
LicensePlateRecognitionPipeline
        |
        v
Plate recognition status
```

## 5. Development scripts

| Script                                         | Responsibility                  |
| ---------------------------------------------- | ------------------------------- |
| `scripts/test_plate_detector.py`             | Detector smoke test             |
| `scripts/test_plate_component.py`            | Detector component test         |
| `scripts/test_plate_pipeline.py`             | Detector-only pipeline test     |
| `scripts/test_ocr.py`                        | Initial PaddleOCR smoke test    |
| `scripts/test_ocr_reader.py`                 | OCR Reader test                 |
| `scripts/test_plate_cropper.py`              | Cropper smoke test              |
| `scripts/test_plate_recognition_pipeline.py` | Detection → Crop → OCR test   |
| `scripts/test_plate_matching.py`             | Plate matching integration test |

Scripts are development tools, not production modules. Later, move meaningful assertions into automated tests and remove redundant scripts.

## 6. Current verified test baseline

Latest reported checks before the matching integration:

- Ruff check: passed.
- Ruff format check: passed.
- Pytest: 10 passed.

OCR integration smoke test:

- Image dimensions: 542 × 372.
- Detected plates: 1.
- OCR output: `29.H1339.79`.
- Normalized output: `29H133979`.
- OCR confidence: 0.9491.

Detector limitation recorded:

- The sample image contains two motorcycles.
- Only one license plate was detected.
- Lowering detector confidence threshold to 0.10 did not recover the second plate.
- Do not change the model based on this single sample; retain this case for end-to-end evaluation.

## 7. Known technical constraints

- Windows environment uses Python 3.12 in `.venv`.
- PyTorch CUDA 12.6 is installed and working on NVIDIA T600 Laptop GPU.
- ONNX Runtime CPU is the current baseline.
- PaddleOCR currently runs on CPU.
- Import `torch` before Paddle/PaddleOCR on Windows.
- Model weights are excluded from Git.
- Do not create a second Python environment unless the current environment is proven unworkable.

## 8. Rules for future tickets

Before sending code:

1. Check this registry.
2. Confirm the target file exists.
3. Confirm the existing class and method names.
4. Reuse existing components.
5. Avoid unnecessary dependency changes.
6. Include the exact import path and correct class name.
7. Provide finite implementation steps and expected output.
8. Define acceptance criteria and tests.
9. Update this registry when interfaces change.

## 9. Pending verification

The following items must be checked against the actual local repository before treating this document as a fully verified API reference:

- Exact class and constructor signature in `recognition_pipeline.py`.
- Exact dataclass and enum names in `recognition_pipeline.py`.
- Whether the matching script imports and initializes the same class name.
- Current contents of all development scripts.
- Whether unit tests exist for detector, cropper, OCR reader, and recognition pipeline.

**Registry status:** Initial draft — partially verified from implementation history and terminal output.
