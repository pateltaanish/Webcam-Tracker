"""Unit tests for webcam_tracker.visualization.recorder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from webcam_tracker.visualization import FrameRecorder


def _frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_write_creates_playable_file_sized_to_first_frame(tmp_path: Path) -> None:
    output_path = tmp_path / "clip.avi"
    recorder = FrameRecorder(output_path, fps=10.0)

    for _ in range(3):
        recorder.write(_frame(width=64, height=48))
    recorder.close()

    assert output_path.exists()
    cap = cv2.VideoCapture(str(output_path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        cap.release()


def test_zero_fps_falls_back_to_default(tmp_path: Path) -> None:
    recorder = FrameRecorder(tmp_path / "clip.avi", fps=0.0)
    assert recorder._fps == 30.0  # noqa: SLF001 -- verifying the fallback, not behavior


def test_close_without_any_writes_is_safe(tmp_path: Path) -> None:
    recorder = FrameRecorder(tmp_path / "clip.avi", fps=10.0)
    recorder.close()  # must not raise even though write() was never called
    assert not (tmp_path / "clip.avi").exists()


def test_close_is_safe_to_call_twice(tmp_path: Path) -> None:
    recorder = FrameRecorder(tmp_path / "clip.avi", fps=10.0)
    recorder.write(_frame())
    recorder.close()
    recorder.close()  # must not raise
