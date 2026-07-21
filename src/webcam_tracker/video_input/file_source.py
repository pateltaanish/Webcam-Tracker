"""Playback from a single recorded video file."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from webcam_tracker.logging_utils import get_logger
from webcam_tracker.video_input.base import FrameSource, VideoFrame, VideoSourceError

logger = get_logger(__name__)


class VideoFileSource(FrameSource):
    """Reads frames from a single video file, in order, once."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cap: cv2.VideoCapture | None = None
        self._fps = 0.0
        self._frame_index = 0

    def open(self) -> None:
        if self._cap is not None:
            logger.warning("open() called while already open; closing previous capture first")
            self.close()
        if not self._path.is_file():
            raise VideoSourceError(f"Video file not found: {self._path}")
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise VideoSourceError(f"Could not open video file: {self._path}")
        self._cap = cap
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        logger.info(
            "Video file opened", extra={"path": str(self._path), "fps": round(self._fps, 1)}
        )

    def read(self) -> VideoFrame | None:
        if self._cap is None:
            raise VideoSourceError("read() called before open()")
        ok, image = self._cap.read()
        if not ok:
            return None  # End of file: expected termination, not an error.
        frame = VideoFrame(
            image=image,
            frame_index=self._frame_index,
            timestamp=time.monotonic(),
            source_fps=self._fps,
        )
        self._frame_index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps
