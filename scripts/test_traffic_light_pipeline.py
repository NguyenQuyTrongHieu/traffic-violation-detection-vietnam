from collections import Counter
from math import ceil, floor
from pathlib import Path

from traffic_violation_detection_vietnam.detection.traffic_light_classifier import (
    TrafficLightStateClassifier,
)
from traffic_violation_detection_vietnam.detection.traffic_light_detector import (
    TrafficLightDetector,
)
from traffic_violation_detection_vietnam.video.video_frame_extractor import (
    VideoFrameExtractor,
)

VIDEO_PATH = Path("scripts/data_test/detect_red_light.mp4")
WEIGHTS_PATH = Path("models/detection/yolo11n.pt")


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    extractor = VideoFrameExtractor(sampling_fps=2.0)
    detector = TrafficLightDetector(
        weights_path=str(WEIGHTS_PATH),
        image_size=640,
        confidence_threshold=0.25,
    )
    classifier = TrafficLightStateClassifier()

    metadata = extractor.get_metadata(str(VIDEO_PATH))

    print("=== VIDEO METADATA ===")
    print(f"Resolution: {metadata.width}x{metadata.height}")
    print(f"Source FPS: {metadata.fps:.2f}")
    print(f"Duration: {metadata.duration_ms / 1000:.2f}s")
    print(f"Sampling FPS: {metadata.effective_sampling_fps:.2f}")
    print(f"Expected sampled frames: {metadata.estimated_sampled_frame_count}")

    total_frames = 0
    frames_with_detections = 0
    total_detections = 0
    state_counts: Counter[str] = Counter()
    detector_times: list[float] = []

    print("\n=== TRAFFIC LIGHT DETECTION ===")

    for frame_data in extractor.iter_frames(str(VIDEO_PATH)):
        total_frames += 1

        detections, inference_time = detector.predict(frame_data.image)
        detector_times.append(inference_time)

        if not detections:
            continue

        frames_with_detections += 1
        total_detections += len(detections)

        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection.bbox_xyxy

            height, width = frame_data.image.shape[:2]
            x1 = max(0, floor(x1))
            y1 = max(0, floor(y1))
            x2 = min(width, ceil(x2))
            y2 = min(height, ceil(y2))

            if x2 <= x1 or y2 <= y1:
                print(f"[{frame_data.timestamp_ms:.0f}ms] Detection {index}: invalid bounding box")
                continue

            crop = frame_data.image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            classification = classifier.predict(crop)
            state = classification.state.value
            state_counts[state] += 1

            print(
                f"[{frame_data.timestamp_ms:8.0f}ms] "
                f"Detection {index} | "
                f"bbox=({x1},{y1},{x2},{y2}) | "
                f"det_conf={detection.confidence:.3f} | "
                f"state={state} | "
                f"color_conf={classification.confidence:.3f} | "
                f"crop={crop.shape[1]}x{crop.shape[0]}"
            )

    print("\n=== SUMMARY ===")
    print(f"Frames processed: {total_frames}")
    print(f"Frames with traffic light: {frames_with_detections}")
    print(f"Total detections: {total_detections}")

    if detector_times:
        average_time = sum(detector_times) / len(detector_times)
        print(f"Average detector inference: {average_time:.2f} ms/frame")

    print("\nClassification counts:")
    for state in ("RED", "YELLOW", "GREEN", "UNKNOWN"):
        print(f"  {state}: {state_counts[state]}")

    if total_detections == 0:
        print("\nWARNING: No traffic lights detected.")
    elif state_counts["UNKNOWN"] > 0:
        print("\nNOTE: Some traffic light crops were classified as UNKNOWN.")


if __name__ == "__main__":
    main()
