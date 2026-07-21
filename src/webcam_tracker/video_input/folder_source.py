"""Plays every video file in a folder back to back, as one continuous source.

Used for batch-testing the pipeline against a whole set of recorded scenarios
(single person, crossing paths, occlusion, etc. — see docs/00_engineering_spec.md
sec 6) without re-running the program per file.
"""

from __future__ import annotations

from pathlib import Path

from webcam_tracker.logging_utils import get_logger
from webcam_tracker.video_input.base import FrameSource, VideoFrame, VideoSourceError
from webcam_tracker.video_input.file_source import VideoFileSource

logger = get_logger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class FolderSource(FrameSource):
    """Concatenates every video file in `folder` (sorted by name) into one stream."""

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._files: list[Path] = []
        self._next_file_index = 0
        self._current: VideoFileSource | None = None
        self._frame_index = 0
        self._fps = 0.0

    def open(self) -> None:
        if not self._folder.is_dir():
            raise VideoSourceError(f"Not a directory: {self._folder}")
        self._files = sorted(
            p for p in self._folder.iterdir() if p.suffix.lower() in _VIDEO_EXTENSIONS
        )
        if not self._files:
            raise VideoSourceError(
                f"No video files found in {self._folder} "
                f"(looked for extensions {sorted(_VIDEO_EXTENSIONS)})"
            )
        logger.info(
            "Folder source opened",
            extra={"folder": str(self._folder), "file_count": len(self._files)},
        )
        self._next_file_index = 0
        if not self._advance_to_next_file():
            raise VideoSourceError(f"Failed to open any file in {self._folder}")

    def _advance_to_next_file(self) -> bool:
        if self._current is not None:
            self._current.close()
            self._current = None
        if self._next_file_index >= len(self._files):
            return False
        path = self._files[self._next_file_index]
        self._next_file_index += 1
        source = VideoFileSource(path)
        source.open()
        self._current = source
        self._fps = source.fps
        return True

    def read(self) -> VideoFrame | None:
        if self._current is None:
            raise VideoSourceError("read() called before open()")
        while True:
            frame = self._current.read()
            if frame is not None:
                # Re-number frame_index to be continuous across files.
                continuous_frame = VideoFrame(
                    image=frame.image,
                    frame_index=self._frame_index,
                    timestamp=frame.timestamp,
                    source_fps=frame.source_fps,
                )
                self._frame_index += 1
                return continuous_frame
            if not self._advance_to_next_file():
                return None

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None

    @property
    def fps(self) -> float:
        return self._fps
