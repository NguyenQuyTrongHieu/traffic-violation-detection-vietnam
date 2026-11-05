# VTVDS — Implementation Registry

**Project:** Vietnam Traffic Violation Detection System
**Document:** Implementation Registry
**Version:** 0.2
**Current series:** T0-007 — OCR Integration
**Latest completed ticket:** T0-007.5 — Plate Matching Integration
**Purpose:** Canonical reference for implemented modules, classes, functions, imports, and interfaces.

---

## 1. Source-of-truth rules

- Actual source code in the Git repository is the final source of truth.
- This registry documents verified public interfaces and component relationships.
- Before proposing code, verify existing class names, function names, constructor arguments, and method signatures.
- Never invent or rename an existing API without explicitly explaining why.
- When an API changes, update this registry in the same ticket.
- Do not create duplicate components that perform the same responsibility.
- Development scripts are not production modules.

## 2. License plate module

Package path:

`src/traffic_violation_detection_vietnam/license_plate/`

### 2.1 `detector.py`

**Dataclass:** `PlateDetection`

Fields:

- `bbox_xyxy: tuple[float, float, float, float]`
- `confidence: float`
- `class_id: int`

**Class:** `LicensePlateDetector`

Constructor:

```python
LicensePlateDetector(
    weights_path: str | Path = "models/license_plate/yolov8n_edge.pt",
    device: str = "0",
    image_size: int = 640,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
)
```

Method:

```python
predict(
    image: np.ndarray,
) -> tuple[list[PlateDetection], float]
```

Responsibilities:

- Load YOLO license plate detection weights.
- Validate a non-empty, three-channel BGR NumPy image.
- Detect license plates.
- Return bounding boxes, confidence, class ID, and inference time in milliseconds.

Windows import requirement:

- Import `torch` before `ultralytics`.

### 2.2 `cropper.py`

**Class:** `LicensePlateCropper`

Constructor:

```python
LicensePlateCropper(padding_ratio: float = 0.05)
```

Method:

```python
crop(
    image: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> np.ndarray
```

Responsibilities:

- Crop a detected license plate from the original BGR image.
- Apply configurable padding.
- Clip coordinates to image boundaries.
- Validate image and bounding box.
- Return an independent cropped image.

### 2.3 `ocr_reader.py`

**Dataclass:** `OCRResult`

Fields:

- `text: str`
- `confidence: float`
- `recognized_texts: list[str]`
- `recognized_scores: list[float]`

**Class:** `LicensePlateOCR`

Constructor:

```python
LicensePlateOCR(device: str = "cpu")
```

Method:

```python
predict(image: np.ndarray) -> OCRResult
```

Responsibilities:

- Recognize text from a cropped license plate using PaddleOCR.
- Combine recognized text segments into one string.
- Calculate the average confidence across recognized segments.
- Return an empty result when no text is recognized.

Current baseline:

- PaddleOCR 3.4.1.
- PP-OCRv5 server detection model.
- English mobile recognition model.
- CPU inference.

Windows import requirement:

- Import `torch` before `paddleocr`.

Important:

- OCR confidence is not a calibrated probability that the complete plate is correct.
- OCR output may contain separators such as dots and hyphens.
- Empty OCR output must be handled.

### 2.4 `normalizer.py`

Function:

```python
normalize_plate_text(text: str) -> str
```

Responsibilities:

- Convert text to uppercase.
- Remove non-alphanumeric characters.
- Preserve Latin letters and digits.
- Raise `TypeError` for non-string input.

Function:

```python
compare_plate_text(
    detected_text: str,
    expected_text: str,
) -> bool
```

Responsibilities:

- Normalize both inputs.
- Return `False` if either normalized value is empty.
- Perform exact comparison.
- Do not use fuzzy matching.

Implementation note:

- The current recognition pipeline calls `normalize_plate_text()` and performs exact equality itself.
- `compare_plate_text()` exists as a standalone helper and is covered by unit tests.

### 2.5 `recognition_pipeline.py`

**Enum:** `PlateMatchStatus`

Values:

- `NOT_CHECKED`
- `PLATE_MATCH`
- `PLATE_MISMATCH`
- `PLATE_UNREADABLE`

**Dataclass:** `PlateRecognition`

Fields:

- `detection: PlateDetection`
- `ocr_result: OCRResult`
- `crop: np.ndarray`
- `normalized_text: str`
- `match_status: PlateMatchStatus`

Property:

```python
is_match -> bool | None
```

Behavior:

- Returns `None` when status is `NOT_CHECKED`.
- Returns `True` when status is `PLATE_MATCH`.
- Returns `False` for other checked statuses.

**Dataclass:** `LicensePlateRecognitionResult`

Fields:

- `recognitions: list[PlateRecognition]`
- `detection_count: int`
- `detection_time_ms: float`
- `total_processing_time_ms: float`
- `image_width: int`
- `image_height: int`
- `annotated_image: np.ndarray`
- `expected_plate: str | None`
- `status: PlateMatchStatus`

**Class:** `LicensePlateRecognitionPipeline`

Constructor:

```python
LicensePlateRecognitionPipeline(
    detector,
    ocr_reader: LicensePlateOCR,
    cropper: LicensePlateCropper | None = None,
)
```

Method:

```python
process(
    image: np.ndarray,
    expected_plate: str | None = None,
) -> LicensePlateRecognitionResult
```

Responsibilities:

1. Validate input image.
2. Normalize the optional expected plate.
3. Run license plate detection.
4. Crop every detected plate.
5. Run OCR on every crop.
6. Normalize recognized text.
7. Compare each recognized plate against the expected plate when provided.
8. Annotate the original image.
9. Aggregate an overall matching status.
10. Return recognition metadata and processing times.

Per-plate status rules:

- No expected plate: `NOT_CHECKED`.
- Empty normalized OCR text: `PLATE_UNREADABLE`.
- Exact normalized match: `PLATE_MATCH`.
- Otherwise: `PLATE_MISMATCH`.

Overall status aggregation:

- No expected plate: `NOT_CHECKED`.
- At least one plate matches: `PLATE_MATCH`.
- No detections, or every detected plate is unreadable: `PLATE_UNREADABLE`.
- Otherwise: `PLATE_MISMATCH`.

Annotation colors:

- Match / not checked: green.
- Mismatch: red.
- Unreadable: orange.

Implementation note:

- The pipeline imports `normalize_plate_text()` directly.
- It does not currently call `compare_plate_text()`.
- The detector argument is not explicitly type-annotated in the constructor.
- The pipeline processes every detection, not only the first one.

## 3. Detector-only pipeline

### `pipeline.py`

**Protocol:** `DetectorProtocol`

Method:

```python
predict(
    image: np.ndarray,
) -> tuple[list[PlateDetection], float]
```

**Dataclass:** `LicensePlatePipelineResult`

Fields:

- `detections: list[PlateDetection]`
- `detection_count: int`
- `inference_time_ms: float`
- `image_width: int`
- `image_height: int`
- `annotated_image: np.ndarray`

**Class:** `LicensePlatePipeline`

Constructor:

```python
LicensePlatePipeline(detector: DetectorProtocol)
```

Method:

```python
process(image: np.ndarray) -> LicensePlatePipelineResult
```

Responsibilities:

- Run plate detection.
- Draw detection bounding boxes and confidence labels.
- Return an annotated image and detection metadata.

This detector-only pipeline is separate from the integrated OCR recognition pipeline.

## 4. Component dependency map

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
Exact comparison with expected plate
        |
        v
PlateRecognition[]
        |
        v
LicensePlateRecognitionResult
```

## 5. Development scripts

| Script                                         | Responsibility                  |
| ---------------------------------------------- | ------------------------------- |
| `scripts/test_plate_detector.py`             | Detector smoke test             |
| `scripts/test_plate_component.py`            | Detector component test         |
| `scripts/test_plate_pipeline.py`             | Detector-only pipeline test     |
| `scripts/test_ocr.py`                        | Initial PaddleOCR smoke test    |
| `scripts/test_ocr_reader.py`                 | OCR reader test                 |
| `scripts/test_plate_cropper.py`              | Cropper smoke test              |
| `scripts/test_plate_recognition_pipeline.py` | Detection → Crop → OCR test   |
| `scripts/test_plate_matching.py`             | Plate matching integration test |

Scripts are development tools, not production modules. Meaningful assertions should progressively move into automated tests.

## 6. Current verified test baseline

Latest reported project-root checks after T0-007.5:

- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv run pytest -q`: 10 passed.

Matching integration smoke test:

- Input image: `scripts/full_LS.png`.
- Expected plate: `29H133979`.
- Detected plates: 1.
- OCR output: `29.H1339.79`.
- Normalized output: `29H133979`.
- OCR confidence: 0.9491.
- Overall status: `PLATE_MATCH`.
- Detection confidence: 0.8632.
- Annotated output: `runs/plate_matching/full_LS_matched.jpg`.

Detector limitation recorded:

- The sample image contains two motorcycles.
- Only one license plate was detected.
- Do not change the model based on this single sample.
- Retain this case for later full-scene evaluation.

Important:

- Smoke tests demonstrate that components run on selected examples.
- They do not establish real-world accuracy or production readiness.

## 7. Known technical constraints

- Windows 11.
- Python 3.12 in `.venv`.
- PyTorch CUDA 12.6 is installed and working on NVIDIA T600 Laptop GPU.
- ONNX Runtime CPU is the current baseline.
- PaddleOCR currently runs on CPU.
- Import `torch` before Paddle/PaddleOCR on Windows.
- Model weights are excluded from Git.
- Do not create a second Python environment unless the current environment is proven unworkable.
- Run project checks from the repository root, not from `scripts/`.

## 8. Rules for future tickets

Before sending code:

1. Check this registry.
2. Confirm the target file exists in GitHub.
3. Confirm existing class and method names.
4. Reuse existing components.
5. Avoid unnecessary dependency changes.
6. Include exact import paths and correct class names.
7. Provide complete file contents when replacing a file.
8. Define finite implementation steps and expected outputs.
9. Define acceptance criteria and tests.
10. Update this registry when interfaces change.

## 9. Remaining limitations and future verification

- No comprehensive OCR accuracy evaluation has been completed.
- No representative full-scene detection benchmark has been completed.
- No video-based violation detection has been implemented yet.
- No automated unit tests currently cover every detector, cropper, OCR, and recognition-pipeline behavior.
- The current matching logic is exact string matching after normalization; it does not correct OCR character confusions.
- Multiple detected plates are aggregated as a match if any one plate matches the expected plate. This behavior may need reconsideration when vehicle-to-plate association is implemented.

**Registry status:** Updated against the source files on `main` after T0-007.5. Smoke-test baseline is verified from user-provided terminal output; broader model accuracy remains unevaluated.
