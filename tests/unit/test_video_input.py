"""Unit tests for the video_input module.

File/folder sources are tested against small synthetic videos generated on
the fly (MJPG/.avi, broadly supported without extra codecs) so these tests
have no dependency on external test data or real camera hardware, and are
therefore safe to run in CI on any machine.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from webcam_tracker.config.settings import AppConfig, LoggingConfig, PathsConfig, VideoConfig
from webcam_tracker.video_input import (
    FolderSource,
    VideoFileSource,
    VideoSourceError,
    WebcamSource,
    create_source,
)


def _write_synthetic_video(path: Path, num_frames: int, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    for i in range(num_frames):
        frame = np.full((48, 64, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_config(source: str) -> AppConfig:
    return AppConfig(  # type: ignore[call-arg]
        video=VideoConfig(source=source, requested_width=64, requested_height=48, requested_fps=10),
        logging=LoggingConfig(),
        paths=PathsConfig(),
    )


class TestVideoFileSource:
    def test_reads_all_frames_then_returns_none(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.avi"
        _write_synthetic_video(video_path, num_frames=5)

        source = VideoFileSource(video_path)
        with source:
            frames = list(source)

        assert len(frames) == 5
        assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4]

    def test_calling_open_twice_is_safe_and_still_readable(self, tmp_path: Path) -> None:
        # Regression test: a caller doing `source.open()` followed by
        # `with source:` (which also calls open()) must not leave the
        # source broken -- this exact bug made the live webcam preview
        # scripts fail with "stopped producing frames" on a real machine.
        video_path = tmp_path / "clip.avi"
        _write_synthetic_video(video_path, num_frames=3)

        source = VideoFileSource(video_path)
        source.open()
        source.open()  # must not raise, leak, or break subsequent reads
        try:
            frames = list(source)
        finally:
            source.close()

        assert len(frames) == 3

    def test_missing_file_raises_on_open(self, tmp_path: Path) -> None:
        source = VideoFileSource(tmp_path / "does_not_exist.avi")
        with pytest.raises(VideoSourceError):
            source.open()

    def test_read_before_open_raises(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.avi"
        _write_synthetic_video(video_path, num_frames=1)
        source = VideoFileSource(video_path)
        with pytest.raises(VideoSourceError):
            source.read()


class TestFolderSource:
    def test_concatenates_files_with_continuous_frame_index(self, tmp_path: Path) -> None:
        _write_synthetic_video(tmp_path / "a.avi", num_frames=3)
        _write_synthetic_video(tmp_path / "b.avi", num_frames=2)

        source = FolderSource(tmp_path)
        with source:
            frames = list(source)

        assert len(frames) == 5
        assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4]

    def test_empty_folder_raises(self, tmp_path: Path) -> None:
        source = FolderSource(tmp_path)
        with pytest.raises(VideoSourceError):
            source.open()

    def test_nonexistent_folder_raises(self, tmp_path: Path) -> None:
        source = FolderSource(tmp_path / "nope")
        with pytest.raises(VideoSourceError):
            source.open()


class TestWebcamSource:
    def test_unreachable_index_raises_on_open(self) -> None:
        # Index 999 has no camera on any real machine -- this exercises the
        # error path deterministically without depending on hardware being
        # present at all.
        source = WebcamSource(
            device_index=999, requested_width=640, requested_height=480, requested_fps=30
        )
        with pytest.raises(VideoSourceError):
            source.open()


class TestCreateSource:
    def test_auto_dispatches_to_webcam_source_with_no_fixed_index(self, tmp_path: Path) -> None:
        config = _make_config("auto")
        source = create_source(config)
        assert isinstance(source, WebcamSource)
        assert source._requested_index is None  # noqa: SLF001 -- verifying dispatch, not behavior

    def test_numeric_dispatches_to_webcam_source_with_fixed_index(self, tmp_path: Path) -> None:
        config = _make_config("2")
        source = create_source(config)
        assert isinstance(source, WebcamSource)
        assert source._requested_index == 2  # noqa: SLF001

    def test_file_path_dispatches_to_video_file_source(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.avi"
        _write_synthetic_video(video_path, num_frames=1)
        config = _make_config(str(video_path))
        source = create_source(config)
        assert isinstance(source, VideoFileSource)

    def test_directory_dispatches_to_folder_source(self, tmp_path: Path) -> None:
        _write_synthetic_video(tmp_path / "clip.avi", num_frames=1)
        config = _make_config(str(tmp_path))
        source = create_source(config)
        assert isinstance(source, FolderSource)

    def test_garbage_source_raises(self, tmp_path: Path) -> None:
        config = _make_config("not-a-real-path-or-index")
        with pytest.raises(VideoSourceError):
            create_source(config)
