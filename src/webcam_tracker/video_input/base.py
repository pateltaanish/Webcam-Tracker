"""Core types shared by every video source implementation.

Why an abstraction at all: the spec requires the same pipeline code to run
against a live webcam, a recorded file, a folder of files, and eventually a
network camera, without caring which one it's pointed at. Every concrete
source below implements this same small interface so `detection`/`tracking`
downstream never need to know or care which one is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


class VideoSourceError(RuntimeError):
    """Raised when a video source can't be opened or read from.

    This is a system boundary (real hardware / real files) where failure is
    expected and must be handled explicitly by the caller, not silently
    swallowed or retried forever.
    """


@dataclass(frozen=True)
class VideoFrame:
    """One captured frame plus the metadata the rest of the pipeline needs.

    `timestamp` uses `time.monotonic()` (seconds since an arbitrary,
    process-local reference point) rather than wall-clock time, because it's
    immune to system clock adjustments — what later modules (stale-frame
    detection, latency measurement, motion prediction) actually care about is
    the *interval* between frames, not the calendar time.
    """

    image: np.ndarray  # BGR, uint8, shape (height, width, 3) — OpenCV's native format
    frame_index: int  # 0-based, increments per frame from this source
    timestamp: float  # time.monotonic() seconds at the moment this frame was read
    source_fps: float  # the source's actual (not requested) frame rate, if known


class FrameSource(ABC):
    """Common interface for webcam / video-file / folder / (later) network sources."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource (camera device, file handle, ...).

        Must raise VideoSourceError on failure rather than leaving the source
        in a half-open state.
        """

    @abstractmethod
    def read(self) -> VideoFrame | None:
        """Return the next frame, or None when the source is exhausted
        (end of file/folder). A live webcam source never returns None on its
        own — only closing it does.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource. Safe to call multiple times."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """The source's actual frame rate, once known (0.0 before open())."""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[VideoFrame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
