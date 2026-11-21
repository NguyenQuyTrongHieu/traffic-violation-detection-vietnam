import sys
from pathlib import Path

from traffic_violation_detection_vietnam.video import VideoFrameExtractor


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: uv run python scripts/test_video_frame_extractor.py <video_path> [sampling_fps]"
        )

    video_path = Path(sys.argv[1])
    sampling_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    extractor = VideoFrameExtractor(sampling_fps=sampling_fps)
    metadata = extractor.get_metadata(video_path)

    print("Video:", metadata.path)
    print("Resolution:", f"{metadata.width} x {metadata.height}")
    print("Source FPS:", metadata.fps)
    print("Source frame count:", metadata.frame_count)
    print("Duration:", round(metadata.duration_ms / 1000, 2), "seconds")
    print("Requested sampling FPS:", metadata.sampling_fps)
    print("Effective sampling FPS:", round(metadata.effective_sampling_fps, 3))
    print("Frame step:", metadata.frame_step)
    print("Estimated sampled frames:", metadata.estimated_sampled_frame_count)

    sampled_count = 0
    first_frame = None
    last_frame = None

    for video_frame in extractor.iter_frames(video_path):
        sampled_count += 1

        if first_frame is None:
            first_frame = video_frame

        last_frame = video_frame

    if first_frame is None or last_frame is None:
        raise RuntimeError("No frames were extracted")

    print("Actual sampled frames:", sampled_count)
    print(
        "First frame:",
        first_frame.frame_index,
        "| timestamp:",
        round(first_frame.timestamp_ms, 2),
        "ms",
        "| shape:",
        first_frame.image.shape,
    )
    print(
        "Last frame:",
        last_frame.frame_index,
        "| timestamp:",
        round(last_frame.timestamp_ms, 2),
        "ms",
        "| shape:",
        last_frame.image.shape,
    )


if __name__ == "__main__":
    main()
