VTVDS — Implementation Registry
Project: Vietnam Traffic Violation Detection System
Document: Implementation Registry
Version: 0.4
Current series: T0-010 — Video Pipeline
Latest completed ticket: T0-009.3 — Violation Logic Review/Bonus
Purpose: Canonical reference for implemented modules, classes, functions, imports, and interfaces.

1. Source-of-truth rules

- Actual source code in the Git repository is the final source of truth.
- This registry documents verified public interfaces and component relationships.
- Before proposing code, verify existing class names, function names, constructor arguments, and method signatures.
- Never invent or rename an existing API without explicitly explaining why.
- When an API changes, update this registry in the same ticket.
- Do not create duplicate components that perform the same responsibility.
- Development scripts are not production modules.
- The test and smoke-test results below are based on the latest user-reported local run; they are not independently executed by this registry update.

2. License plate module
   Package path:

src/traffic_violation_detection_vietnam/license_plate/
2.1 detector.py
Dataclass: PlateDetection

Fields:

- bbox_xyxy: tuple[float, float, float, float]
- confidence: float
- class_id: int
  Class: LicensePlateDetector

Constructor:
LicensePlateDetector(
    weights_path: str | Path = "models/license_plate/yolov8n_edge.pt",
    device: str = "0",
    image_size: int = 640,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
)
Method:
predict(
    image: np.ndarray,
) -> tuple[list[PlateDetection], float]
Responsibilities:

- Load YOLO license plate detection weights.
- Validate a non-empty, three-channel BGR NumPy image.
- Detect license plates.
- Return bounding boxes, confidence, class ID, and inference time in milliseconds.
  Windows import requirement:
- Import torch before ultralytics.
  2.2 cropper.py
  Class: LicensePlateCropper

Constructor:
LicensePlateCropper(padding_ratio: float = 0.05)
Method:
crop(
    image: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> np.ndarray
Responsibilities:

- Crop a detected license plate from the original BGR image.
- Apply configurable padding.
- Clip coordinates to image boundaries.
- Validate image and bounding box.
- Return an independent cropped image.
  2.3 ocr_reader.py
  Dataclass: OCRResult

Fields:

- text: str
- confidence: float
- recognized_texts: list[str]
- recognized_scores: list[float]
  Class: LicensePlateOCR

Constructor:
LicensePlateOCR(device: str = "cpu")
Method:
predict(image: np.ndarray) -> OCRResult
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
- Import torch before paddleocr.
  Important:
- OCR confidence is not a calibrated probability that the complete plate is correct.
- OCR output may contain separators such as dots and hyphens.
- Empty OCR output must be handled.
  2.4 normalizer.py
  Function:
  normalize_plate_text(text: str) -> str
  Responsibilities:
- Convert text to uppercase.
- Remove non-alphanumeric characters.
- Preserve Latin letters and digits.
- Raise TypeError for non-string input.
  Function:
  compare_plate_text(
  detected_text: str,
  expected_text: str,
  ) -> bool
  Responsibilities:
- Normalize both inputs.
- Return False if either normalized value is empty.
- Perform exact comparison.
- Do not use fuzzy matching.
  Implementation note:
- The current recognition pipeline calls normalize_plate_text() and performs exact equality itself.
- compare_plate_text() exists as a standalone helper and is covered by unit tests.
  2.5 recognition_pipeline.py
  Enum: PlateMatchStatus

Values:

- NOT_CHECKED
- PLATE_MATCH
- PLATE_MISMATCH
- PLATE_UNREADABLE
  Dataclass: PlateRecognition

Fields:

- detection: PlateDetection
- ocr_result: OCRResult
- crop: np.ndarray
- normalized_text: str
- match_status: PlateMatchStatus
  Property:
  is_match -> bool | None
  Behavior:
- Returns None when status is NOT_CHECKED.
- Returns True when status is PLATE_MATCH.
- Returns False for other checked statuses.
  Dataclass: LicensePlateRecognitionResult

Fields:

- recognitions: list[PlateRecognition]
- detection_count: int
- detection_time_ms: float
- total_processing_time_ms: float
- image_width: int
- image_height: int
- annotated_image: np.ndarray
- expected_plate: str | None
- status: PlateMatchStatus
  Class: LicensePlateRecognitionPipeline

Constructor:
LicensePlateRecognitionPipeline(
    detector,
    ocr_reader: LicensePlateOCR,
    cropper: LicensePlateCropper | None = None,
)
Method:
process(
    image: np.ndarray,
    expected_plate: str | None = None,
) -> LicensePlateRecognitionResult
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

- No expected plate: NOT_CHECKED.
- Empty normalized OCR text: PLATE_UNREADABLE.
- Exact normalized match: PLATE_MATCH.
- Otherwise: PLATE_MISMATCH.
  Overall status aggregation:
- No expected plate: NOT_CHECKED.
- At least one plate matches: PLATE_MATCH.
- No detections, or every detected plate is unreadable: PLATE_UNREADABLE.
- Otherwise: PLATE_MISMATCH.
  Annotation colors:
- Match / not checked: green.
- Mismatch: red.
- Unreadable: orange.
  Implementation note:
- The pipeline imports normalize_plate_text() directly.
- It does not currently call compare_plate_text().
- The detector argument is not explicitly type-annotated in the constructor.
- The pipeline processes every detection, not only the first one.

3. Detector-only license plate pipeline
   pipeline.py
   Protocol: DetectorProtocol

Method:
predict(
    image: np.ndarray,
) -> tuple[list[PlateDetection], float]
Dataclass: LicensePlatePipelineResult

Fields:

- detections: list[PlateDetection]
- detection_count: int
- inference_time_ms: float
- image_width: int
- image_height: int
- annotated_image: np.ndarray
  Class: LicensePlatePipeline

Constructor:
LicensePlatePipeline(detector: DetectorProtocol)
Method:
process(image: np.ndarray) -> LicensePlatePipelineResult
Responsibilities:

- Run plate detection.
- Draw detection bounding boxes and confidence labels.
- Return an annotated image and detection metadata.
  This detector-only pipeline is separate from the integrated OCR recognition pipeline.

4. Person and vehicle detection
   Package path:

src/traffic_violation_detection_vietnam/detection/
4.1 person_vehicle_detector.py
Dataclass: ObjectDetection

Fields:

- bbox_xyxy
- confidence
- class_id
- class_name
  Class: PersonVehicleDetector

Constructor:
PersonVehicleDetector(
    weights_path="models/detection/yolo11n.pt",
    device=None,
    image_size=640,
    confidence_threshold=0.25,
    iou_threshold=0.45,
)
Method:
predict(
    image: np.ndarray,
) -> tuple[list[ObjectDetection], float]
Responsibilities:

- Load the pretrained YOLO11n COCO model.
- Detect and retain COCO class 0 (person) and class 3 (motorcycle).
- Return ObjectDetection records and inference time in milliseconds.
  Implementation notes:
- Model weights are stored locally at models/detection/yolo11n.pt and excluded from Git.
- Import torch before ultralytics on Windows.
- This module detects people and motorcycles; it does not identify riders or decide violations.
  4.2 helmet_detector.py
  Dataclass: HelmetDetection

Fields:

- bbox_xyxy
- confidence
- class_id
- class_name
  Class: HelmetDetector

Constructor:
HelmetDetector(
    weights_path="models/detection/helmet_yolo11s.pt",
    device=None,
    image_size=640,
    confidence_threshold=0.25,
    iou_threshold=0.45,
)
Method:
predict(
    image: np.ndarray,
) -> tuple[list[HelmetDetection], float]
Responsibilities:

- Load the selected pretrained YOLO11 helmet detector.
- Detect helmet-related objects.
- Map class ID 0 to with_helmet.
- Map class ID 1 to without_helmet.
- Return HelmetDetection records and inference time in milliseconds.
  Model and quality notes:
- Current checkpoint: models/detection/helmet_yolo11s.pt (local; excluded from Git).
- The model card reports validation metrics on a small validation set; these are author-reported and are not independent evidence of real-world performance.
- Current tests show view-angle sensitivity and weak generalization for the without_helmet class.
- In some front/high-angle views, no-helmet heads were missed; in some side views with the face visible, no-helmet cases were classified as with_helmet.
- Treat this detector as a development baseline, not as authoritative violation evidence.
- Fine-tuning and representative evaluation are a separate parallel workstream; they do not block development of the end-to-end pipeline.
- Preserve the detector interface when replacing weights, and re-run regression tests and evaluation.

5. Person–motorcycle–helmet association
   Package path:

src/traffic_violation_detection_vietnam/association/
person_vehicle_helmet_associator.py
Enum: AssociationStatus

Values:

- ASSOCIATED
- NOT_ASSOCIATED
- AMBIGUOUS
- NOT_DETECTED
  Dataclass: PersonAssociation

Fields:

- person
- motorcycle
- helmet
- motorcycle_status
- helmet_status
- motorcycle_score
- helmet_score
  Dataclass: PersonVehicleHelmetAssociationResult

Fields:

- associations
- unassigned_helmets
- ambiguous_helmets
- unassigned_motorcycles
  Class: PersonVehicleHelmetAssociator

Constructor:
PersonVehicleHelmetAssociator(
    head_region_ratio=0.45,
    lower_body_start_ratio=0.45,
    min_helmet_coverage=0.30,
    min_vehicle_overlap=0.15,
    ambiguity_margin=0.10,
)
Method:
associate(
    persons,
    motorcycles,
    helmets,
) -> PersonVehicleHelmetAssociationResult
Responsibilities and current heuristic:

- Associate each person detection with a motorcycle using overlap between the motorcycle box and the person's lower-body region.
- Associate helmet detections with person head regions using helmet-box coverage.
- Do not force an assignment when candidates are ambiguous.
- Assign a helmet to at most one person.
- Allow one motorcycle to be associated with multiple people, because multiple riders may share one motorcycle.
- Return unassigned and ambiguous detections for downstream handling.
  Status semantics:
- ASSOCIATED: an association was made.
- NOT_ASSOCIATED: no suitable association was found.
- AMBIGUOUS: multiple candidates are too close to choose safely.
- NOT_DETECTED: the relevant object (for example, a helmet detection) is absent.
  Critical decision boundary:
- helmet_status == NOT_DETECTED does not mean without_helmet.
- The associator only links detector outputs. It does not make the final violation decision.
- Bounding-box overlap is a geometric heuristic, not proof of physical identity or rider ownership.
- Thresholds have not yet been validated on a representative traffic dataset.

6. Component dependency map
   Image
   |
   +--> PersonVehicleDetector --> persons[], motorcycles[]
   |
   +--> HelmetDetector ---------> helmets[]
   |                                  |
   |                                  v
   |                    PersonVehicleHelmetAssociator
   |                                  |
   |                                  v
   |                         PersonAssociation[]
   |
   +--> LicensePlateDetector --> PlateDetection[]
   |
   v
   LicensePlateCropper
   |
   v
   LicensePlateOCR
   |
   v
   normalize_plate_text()
   |
   v
   Exact expected-plate matching
   |
   v
   PlateRecognition / Result
   Detection and association outputs feed the T0-009 violation decision layer. The current decision modules consume structured evidence; video ingestion, traffic-light recognition, tracking, and stop-line crossing extraction are not yet implemented.

6.1 Violation decision logic
   Package path:

src/traffic_violation_detection_vietnam/violations/

6.1.1 helmet_violation_decider.py
Enums:
- HelmetViolationStatus: VIOLATION_CANDIDATE, NO_VIOLATION_DETECTED, NEEDS_REVIEW, NOT_APPLICABLE.
- HelmetViolationReason: RIDER_WITHOUT_HELMET, RIDER_WITH_HELMET, MOTORCYCLE_NOT_ASSOCIATED, MOTORCYCLE_ASSOCIATION_AMBIGUOUS, MOTORCYCLE_DATA_MISSING, HELMET_NOT_DETECTED, HELMET_ASSOCIATION_AMBIGUOUS, HELMET_DATA_INCONSISTENT, HELMET_CONFIDENCE_INVALID, HELMET_CONFIDENCE_TOO_LOW, UNKNOWN_HELMET_CLASS, MOTORCYCLE_DATA_INCONSISTENT, HELMET_CLASS_INCONSISTENT.

Dataclass: HelmetViolationDecision
Fields:
- person: ObjectDetection
- motorcycle: ObjectDetection | None
- helmet: HelmetDetection | None
- status: HelmetViolationStatus
- reason_code: HelmetViolationReason
- helmet_confidence: float | None

Class: HelmetViolationDecider
Constructor:
HelmetViolationDecider(min_helmet_confidence: float = 0.5)
Method:
decide(association_result: PersonVehicleHelmetAssociationResult) -> list[HelmetViolationDecision]

Decision behavior:
- Evaluate each person association independently.
- A person without an associated motorcycle is NOT_APPLICABLE.
- Inconsistent motorcycle status/object combinations require NEEDS_REVIEW.
- Ambiguous motorcycle or helmet association requires NEEDS_REVIEW.
- A missing helmet detection is not interpreted as a no-helmet violation; it requires NEEDS_REVIEW.
- Validate helmet class ID/name consistency: class 0 = with_helmet; class 1 = without_helmet.
- Unknown or inconsistent class metadata requires NEEDS_REVIEW.
- Invalid or low helmet confidence requires NEEDS_REVIEW.
- A valid without_helmet classification returns VIOLATION_CANDIDATE.
- A valid with_helmet classification returns NO_VIOLATION_DETECTED.
- The default 0.5 confidence threshold is provisional and not calibrated.

6.1.2 red_light_violation_decider.py
Enums:
- TrafficLightState: RED, YELLOW, GREEN, UNKNOWN.
- CrossingStatus: CROSSED, NOT_CROSSED, AMBIGUOUS, NOT_EVALUABLE.
- RedLightViolationStatus: VIOLATION_CANDIDATE, NO_VIOLATION_DETECTED, NEEDS_REVIEW, NOT_APPLICABLE.
- RedLightViolationReason: MOTORCYCLE_CROSSED_ON_RED, MOTORCYCLE_CROSSED_ON_GREEN, MOTORCYCLE_DID_NOT_CROSS, NON_MOTORCYCLE, CROSSING_AMBIGUOUS, CROSSING_NOT_EVALUABLE, INVALID_CROSSING_EVIDENCE, TRACK_ID_MISSING, TRAFFIC_LIGHT_OBSERVATION_MISSING, TRAFFIC_LIGHT_TIME_GAP_TOO_LARGE, TRAFFIC_LIGHT_STATE_TRANSITION, TRAFFIC_LIGHT_STATE_UNKNOWN, TRAFFIC_LIGHT_YELLOW, TRAFFIC_LIGHT_CONFIDENCE_TOO_LOW, CROSSING_CONFIDENCE_TOO_LOW, INVALID_TRAFFIC_LIGHT_OBSERVATION.

Dataclasses:
- StopLineCrossingEvent(track_id: str, vehicle_class: str, timestamp_ms: float, status: CrossingStatus, confidence: float)
- TrafficLightObservation(timestamp_ms: float, state: TrafficLightState, confidence: float)
- RedLightEvidence(crossing_event: StopLineCrossingEvent, traffic_light_observations: list[TrafficLightObservation])
- RedLightViolationDecision(track_id, status, reason_code, crossing_timestamp_ms, traffic_light_state, traffic_light_confidence)

Class: RedLightViolationDecider
Constructor:
RedLightViolationDecider(min_crossing_confidence: float = 0.5, min_traffic_light_confidence: float = 0.5, max_signal_gap_ms: float = 500.0)
Method:
decide(evidence: RedLightEvidence) -> RedLightViolationDecision

Decision behavior:
- Only motorcycle events are applicable.
- NOT_CROSSED returns NO_VIOLATION_DETECTED.
- AMBIGUOUS or NOT_EVALUABLE crossing returns NEEDS_REVIEW.
- A crossed event requires a valid track ID, timestamp, and confidence.
- Traffic-light observations must be valid and temporally support the crossing timestamp.
- An exact-timestamp observation can be used directly; otherwise, before/after observations must be within max_signal_gap_ms and agree on state.
- RED returns VIOLATION_CANDIDATE; GREEN returns NO_VIOLATION_DETECTED.
- YELLOW, UNKNOWN, state transitions, missing/invalid observations, low confidence, or excessive time gaps return NEEDS_REVIEW.
- Default thresholds and the 500 ms maximum signal gap are provisional, not calibrated.

6.1.3 violations/__init__.py
Exports the public enums, dataclasses, and decider classes listed above.

Decision boundary:
- These modules interpret structured detections/associations/evidence; they do not perform image/video detection, tracking, traffic-light recognition, or stop-line geometry.
- VIOLATION_CANDIDATE is not a legally authoritative enforcement decision.
- Uncertain or inconsistent evidence must remain NEEDS_REVIEW.

7. Development scripts
   Script	Responsibility
   scripts/test_plate_detector.py	License plate detector smoke test
   scripts/test_plate_component.py	License plate detector component test
   scripts/test_plate_pipeline.py	License plate detector-only pipeline test
   scripts/test_ocr.py	Initial PaddleOCR smoke test
   scripts/test_ocr_reader.py	OCR reader test
   scripts/test_plate_cropper.py	Plate cropper smoke test
   scripts/test_plate_recognition_pipeline.py	Detection → Crop → OCR test
   scripts/test_plate_matching.py	Plate matching integration test
   scripts/test_person_vehicle_detector.py	Person/motorcycle detector smoke test
   scripts/test_helmet_detector.py	Helmet detector smoke test
   scripts/test_person_vehicle_helmet_association.py	Combined detector + association smoke test

Scripts are development tools, not production modules. Meaningful assertions should progressively move into automated tests.
8. Current reported test baseline
Latest user-reported project-root checks after T0-009.3 (2026-11-17):

- uv run ruff check .: passed.
- uv run ruff format --check .: passed.
- uv run pytest -q: 65 passed.
  These are user-reported local results; this registry update does not claim an independent test execution.
  8.1 Person/motorcycle/helmet association smoke test
  Script:

scripts/test_person_vehicle_helmet_association.py

User-reported run summary:

- Image size: 962 x 510.
- Person detections: 7.
- Motorcycle detections: 5.
- Helmet detections: 6.
- Vehicle associations: 7.
- Ambiguous vehicle associations: 0.
- Helmet associations: 5.
- Ambiguous helmet associations: 0.
- Unassigned motorcycles: 0.
- Unassigned helmets: 1.
- Person/vehicle inference: 2079.98 ms.
- Helmet inference: 146.38 ms.
- Annotated output: runs/person_vehicle_helmet_association/detection_acsocia_associated.jpg.
  Interpretation and limits:
- Seven person-to-motorcycle associations with five motorcycles is possible because one motorcycle may carry multiple people.
- Five helmet associations for seven people means two people did not receive a helmet assignment in this run. It does not prove those people were not wearing helmets.
- One unassigned helmet may be a false detection or may not overlap a detected head region sufficiently; the count alone cannot determine which.
- No ambiguous associations occurred in this sample.
- The user visually inspected this sample and reported that all three detectors performed well on it.
- This is one smoke-test sample, not a representative accuracy evaluation or throughput benchmark.
  8.2 Plate matching integration smoke test
- Input image: scripts/full_LS.png.
- Expected plate: 29H133979.
- Detected plates: 1.
- OCR output: 29.H1339.79.
- Normalized output: 29H133979.
- OCR confidence: 0.9491.
- Overall status: PLATE_MATCH.
- Detection confidence: 0.8632.
- Annotated output: runs/plate_matching/full_LS_matched.jpg.
  Detector limitation recorded:
- The sample image contains two motorcycles.
- Only one license plate was detected.
- Do not change the model based on this single sample.
- Retain this case for later full-scene evaluation.
  Important:
- Smoke tests demonstrate that components run on selected examples.
- They do not establish real-world accuracy or production readiness.

9. Known technical constraints

- Windows 11.
- Python 3.12 in .venv.
- PyTorch CUDA 12.6 is installed and working on NVIDIA T600 Laptop GPU.
- ONNX Runtime CPU is the current baseline.
- PaddleOCR currently runs on CPU.
- Import torch before Paddle/PaddleOCR and before Ultralytics on Windows.
- Model weights are excluded from Git.
- Do not create a second Python environment unless the current environment is proven unworkable.
- Run project checks from the repository root, not from scripts/.

10. Rules for future tickets
    Before sending code:
11. Check this registry.
12. Confirm the target file exists in GitHub.
13. Confirm existing class and method names.
14. Reuse existing components.
15. Avoid unnecessary dependency changes.
16. Include exact import paths and correct class names.
17. Provide complete file contents when replacing a file.
18. Define finite implementation steps and expected outputs.
19. Define acceptance criteria and tests.
20. Update this registry when interfaces change.
21. Remaining limitations and future verification

- No comprehensive OCR accuracy evaluation has been completed.
- No representative full-scene detection benchmark has been completed.
- No video-based violation detection has been implemented yet; T0-009 decision modules currently consume structured evidence rather than extracting it from frames.
- No automated unit tests currently cover every detector, cropper, OCR, and recognition-pipeline behavior.
- The current matching logic is exact string matching after normalization; it does not correct OCR character confusions.
- Multiple detected plates are aggregated as a match if any one plate matches the expected plate. This behavior may need reconsideration when vehicle-to-plate association is implemented.
- Helmet detection has demonstrated view-angle sensitivity and weak without_helmet generalization on user-tested examples.
- Person/motorcycle/helmet association is based on bounding-box geometry; it is not tracking and does not prove real-world identity.
- Association thresholds have not been calibrated on a representative dataset.
- No representative end-to-end evaluation has been performed on full-scene traffic images/videos.
- Do not equate NOT_DETECTED with WITHOUT_HELMET; an absent detection may result from occlusion, scale, angle, image quality, or model failure.
- Do not use the current helmet model as the sole basis for an authoritative violation decision.
  Registry status: Updated to reflect T0-008.1–T0-008.4 and T0-009.1–T0-009.3 based on GitHub main source review and user-reported local test results available on 2026-11-17. T0-009 is closed; decision logic is implemented, while upstream video evidence extraction and model-quality limitations remain explicit backlog items. Current planned series: T0-010 — Video Pipeline.
