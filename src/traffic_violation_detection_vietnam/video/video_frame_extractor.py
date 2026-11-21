from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_ms: float
    sampling_fps: float
    frame_step: int
    effective_sampling_fps: float
    estimated_sampled_frame_count: int


@dataclass
class VideoFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray


class VideoFrameExtractor:
    def __init__(self, sampling_fps: float = 2.0) -> None:
        if (
            not isinstance(sampling_fps, (int, float))
            or isinstance(sampling_fps, bool)
            or not isfinite(sampling_fps)
            or sampling_fps <= 0
        ):
            raise ValueError("sampling_fps must be a finite number greater than zero")

        self.sampling_fps = float(sampling_fps)

    def _open_video(
        self,
        video_path: str | Path,
    ) -> tuple[cv2.VideoCapture, VideoMetadata]:
        path = Path(video_path)

        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot open video: {path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            values = (fps, frame_count_value, width_value, height_value)

            if not all(isfinite(value) for value in values):
                raise ValueError("Video metadata contains invalid numeric values")

            if fps <= 0:
                raise ValueError("Video FPS must be greater than zero")

            frame_count = int(round(frame_count_value))
            width = int(round(width_value))
            height = int(round(height_value))

            if frame_count <= 0:
                raise ValueError("Video contains no frames or has invalid frame count")

            if width <= 0 or height <= 0:
                raise ValueError("Video has invalid frame dimensions")

            frame_step = max(1, round(fps / self.sampling_fps))
            effective_sampling_fps = fps / frame_step

            metadata = VideoMetadata(
                path=path,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration_ms=frame_count / fps * 1000,
                sampling_fps=self.sampling_fps,
                frame_step=frame_step,
                effective_sampling_fps=effective_sampling_fps,
                estimated_sampled_frame_count=ceil(frame_count / frame_step),
            )

            return capture, metadata

        except Exception:
            capture.release()
            raise

    def get_metadata(self, video_path: str | Path) -> VideoMetadata:
        capture, metadata = self._open_video(video_path)

        try:
            return metadata
        finally:
            capture.release()

    def iter_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        capture, metadata = self._open_video(video_path)
        decoded_frames = 0
        frame_index = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                if (
                    not isinstance(frame, np.ndarray)
                    or frame.size == 0
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                ):
                    raise ValueError(f"Invalid decoded frame at index {frame_index}")

                decoded_frames += 1

                if frame_index % metadata.frame_step == 0:
                    yield VideoFrame(
                        frame_index=frame_index,
                        timestamp_ms=frame_index / metadata.fps * 1000,
                        image=frame,
                    )

                frame_index += 1

            if decoded_frames == 0:
                raise ValueError(f"Video contains no decodable frames: {metadata.path}")

        finally:
            capture.release()
