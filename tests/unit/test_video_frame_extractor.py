import cv2
import numpy as np
import pytest

from traffic_violation_detection_vietnam.video import (
    VideoFrameExtractor,
)


class FakeCapture:
    def __init__(
        self,
        *,
        opened=True,
        fps=30.0,
        frame_count=31,
        width=640,
        height=480,
        frames=None,
    ):
        self.opened = opened
        self.properties = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }
        self.frames = (
            frames
            if frames is not None
            else [
                np.full((height, width, 3), index, dtype=np.uint8) for index in range(frame_count)
            ]
        )
        self.position = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        return self.properties[property_id]

    def read(self):
        if self.position >= len(self.frames):
            return False, None

        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


def create_extractor(monkeypatch, tmp_path, **capture_kwargs):
    video_path = tmp_path / "sample.mp4"
    video_path.touch()

    captures = []

    def fake_video_capture(_path):
        capture = FakeCapture(**capture_kwargs)
        captures.append(capture)
        return capture

    monkeypatch.setattr(
        "traffic_violation_detection_vietnam.video.video_frame_extractor.cv2.VideoCapture",
        fake_video_capture,
    )

    return video_path, captures


def test_get_metadata_returns_video_properties(monkeypatch, tmp_path):
    video_path, captures = create_extractor(monkeypatch, tmp_path)

    metadata = VideoFrameExtractor(sampling_fps=2).get_metadata(video_path)

    assert metadata.fps == 30.0
    assert metadata.frame_count == 31
    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.duration_ms == pytest.approx(1033.33, abs=0.01)
    assert metadata.frame_step == 15
    assert metadata.effective_sampling_fps == 2.0
    assert metadata.estimated_sampled_frame_count == 3
    assert captures[0].released


def test_iter_frames_samples_frames_and_preserves_timestamps(
    monkeypatch,
    tmp_path,
):
    video_path, captures = create_extractor(monkeypatch, tmp_path)

    frames = list(VideoFrameExtractor(sampling_fps=2).iter_frames(video_path))

    assert [frame.frame_index for frame in frames] == [0, 15, 30]
    assert [frame.timestamp_ms for frame in frames] == pytest.approx([0.0, 500.0, 1000.0])
    assert all(frame.image.shape == (480, 640, 3) for frame in frames)
    assert captures[0].released


def test_sampling_fps_higher_than_source_fps_reads_every_frame(
    monkeypatch,
    tmp_path,
):
    video_path, _ = create_extractor(
        monkeypatch,
        tmp_path,
        fps=5.0,
        frame_count=3,
    )

    frames = list(VideoFrameExtractor(sampling_fps=10).iter_frames(video_path))

    assert [frame.frame_index for frame in frames] == [0, 1, 2]


@pytest.mark.parametrize(
    "sampling_fps",
    [0, -1, float("nan"), float("inf"), True],
)
def test_invalid_sampling_fps_raises_error(sampling_fps):
    with pytest.raises(ValueError, match="sampling_fps"):
        VideoFrameExtractor(sampling_fps=sampling_fps)


def test_missing_video_raises_file_not_found(tmp_path):
    extractor = VideoFrameExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.get_metadata(tmp_path / "missing.mp4")


def test_video_that_cannot_be_opened_raises_error(monkeypatch, tmp_path):
    video_path, captures = create_extractor(
        monkeypatch,
        tmp_path,
        opened=False,
    )

    with pytest.raises(ValueError, match="Cannot open video"):
        VideoFrameExtractor().get_metadata(video_path)

    assert captures[0].released


def test_invalid_video_metadata_raises_error(monkeypatch, tmp_path):
    video_path, captures = create_extractor(
        monkeypatch,
        tmp_path,
        fps=0,
    )

    with pytest.raises(ValueError, match="FPS"):
        VideoFrameExtractor().get_metadata(video_path)

    assert captures[0].released


def test_video_without_decodable_frames_raises_error(monkeypatch, tmp_path):
    video_path, captures = create_extractor(
        monkeypatch,
        tmp_path,
        frames=[],
    )

    with pytest.raises(ValueError, match="no decodable frames"):
        list(VideoFrameExtractor().iter_frames(video_path))

    assert captures[0].released


def test_capture_is_released_when_iteration_stops_early(
    monkeypatch,
    tmp_path,
):
    video_path, captures = create_extractor(monkeypatch, tmp_path)

    frame_iterator = VideoFrameExtractor().iter_frames(video_path)
    next(frame_iterator)
    frame_iterator.close()

    assert captures[0].released
