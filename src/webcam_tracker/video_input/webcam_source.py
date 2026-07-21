"""Live webcam capture.

Machine portability: a USB webcam on a desktop and a laptop's built-in
webcam are indistinguishable to OpenCV except for *which device index* the
OS happens to assign them — and that assignment is driver/OS-dependent, not
something we can hardcode once and expect to work everywhere. Two things
handle this:

1. ``device_index=None`` ("auto" in config) probes indices in order and
   uses the first one that actually produces a frame — the common case on
   both a desktop with one USB webcam and a laptop with one built-in webcam,
   where there's exactly one real camera to find.
2. If a machine has more than one camera (e.g. a laptop with its built-in
   cam *and* a USB webcam plugged in) and auto-detection picks the wrong
   one, ``scripts/list_cameras.py`` enumerates every working index so you
   can pin the right one explicitly via ``WEBCAM_TRACKER_VIDEO__SOURCE`` in
   a local ``.env`` (already gitignored, so each machine can have its own).

Windows-specific note: OpenCV's default backend (Media Foundation, MSMF) is
known to hang or open slowly with some USB webcams. We try DirectShow
(``CAP_DSHOW``) first on Windows, then fall back to MSMF, then whatever
OpenCV picks automatically — the same probing logic used for auto-detection.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2

from webcam_tracker.logging_utils import get_logger
from webcam_tracker.video_input.base import FrameSource, VideoFrame, VideoSourceError

logger = get_logger(__name__)

_MAX_CONSECUTIVE_READ_FAILURES = 30  # ~1 second of dropped frames at 30 FPS

BACKEND_NAMES = {
    cv2.CAP_DSHOW: "DSHOW",
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_V4L2: "V4L2",
    cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
    cv2.CAP_ANY: "ANY",
}


def _candidate_backends() -> list[int]:
    """Backends to try, in order, for this platform. CAP_ANY is always the
    last resort — it lets OpenCV pick whatever it thinks is best."""
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def _try_open(index: int, backends: Sequence[int]) -> tuple[cv2.VideoCapture, int] | None:
    """Try each backend for `index` until one opens *and* yields a real frame.

    A backend can report `isOpened() == True` for an index with no real
    camera behind it (common with virtual-camera drivers), so we don't trust
    isOpened() alone — we require one successful frame read too.
    """
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _frame = cap.read()
            if ok:
                return cap, backend
        cap.release()
    return None


@dataclass(frozen=True)
class CameraProbeResult:
    """Outcome of probing one device index. Used by scripts/list_cameras.py."""

    index: int
    working: bool
    backend: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def probe_cameras(max_probe: int = 10) -> list[CameraProbeResult]:
    """Probe every index in 0..max_probe-1 and report which ones work.

    Unlike discover_webcam_index (which stops at the first match), this
    scans every index so a human can see the full picture on a machine with
    multiple cameras.
    """
    results: list[CameraProbeResult] = []
    for index in range(max_probe):
        found = _try_open(index, _candidate_backends())
        if found is None:
            results.append(CameraProbeResult(index=index, working=False))
            continue
        cap, backend = found
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        results.append(
            CameraProbeResult(
                index=index,
                working=True,
                backend=backend,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                # Some drivers report 0 or a negative sentinel when FPS isn't queryable.
                fps=reported_fps if reported_fps > 0 else None,
            )
        )
        cap.release()
    return results


def discover_webcam_index(max_probe: int = 5) -> int:
    """Find the first working webcam index by probing 0..max_probe-1.

    Raises VideoSourceError if none of the probed indices produce a frame.
    """
    for index in range(max_probe):
        result = _try_open(index, _candidate_backends())
        if result is not None:
            cap, _backend = result
            cap.release()
            return index
    raise VideoSourceError(
        f"No working webcam found by probing indices 0..{max_probe - 1}. "
        "Run `.venv\\Scripts\\python.exe scripts\\list_cameras.py` for details, "
        "or set WEBCAM_TRACKER_VIDEO__SOURCE to an explicit index."
    )


class WebcamSource(FrameSource):
    """Captures frames from a live webcam via OpenCV's VideoCapture."""

    def __init__(
        self,
        device_index: int | None,
        requested_width: int,
        requested_height: int,
        requested_fps: int,
        max_probe: int = 5,
    ) -> None:
        """device_index=None triggers auto-detection on open() (see module docstring)."""
        self._requested_index = device_index
        self._requested_width = requested_width
        self._requested_height = requested_height
        self._requested_fps = requested_fps
        self._max_probe = max_probe
        self._cap: cv2.VideoCapture | None = None
        self._resolved_index: int | None = None
        self._actual_fps = 0.0
        self._frame_index = 0

    def open(self) -> None:
        if self._cap is not None:
            # Opening twice would create a second capture handle on the same
            # device while the first is still held -- on Windows/DirectShow
            # that makes both handles fail to read. Close the old one first.
            logger.warning("open() called while already open; closing previous capture first")
            self.close()

        index = (
            self._requested_index
            if self._requested_index is not None
            else discover_webcam_index(self._max_probe)
        )
        result = _try_open(index, _candidate_backends())
        if result is None:
            raise VideoSourceError(
                f"Could not open webcam at index {index} with any known backend."
            )
        cap, backend = result

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        cap.set(cv2.CAP_PROP_FPS, self._requested_fps)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        self._cap = cap
        self._resolved_index = index
        # Some backends/cameras report 0 for FPS even though capture works fine.
        self._actual_fps = actual_fps if actual_fps > 0 else float(self._requested_fps)

        logger.info(
            "Webcam opened",
            extra={
                "device_index": index,
                "backend": backend,
                "requested_resolution": f"{self._requested_width}x{self._requested_height}",
                "requested_fps": self._requested_fps,
                "actual_resolution": f"{actual_width}x{actual_height}",
                "actual_fps": round(self._actual_fps, 1),
            },
        )

    def read(self) -> VideoFrame | None:
        if self._cap is None:
            raise VideoSourceError("read() called before open()")

        failures = 0
        while True:
            ok, image = self._cap.read()
            if ok:
                frame = VideoFrame(
                    image=image,
                    frame_index=self._frame_index,
                    timestamp=time.monotonic(),
                    source_fps=self._actual_fps,
                )
                self._frame_index += 1
                return frame

            failures += 1
            logger.warning(
                "Webcam frame read failed, retrying",
                extra={"device_index": self._resolved_index, "consecutive_failures": failures},
            )
            if failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                raise VideoSourceError(
                    f"Webcam at index {self._resolved_index} stopped producing frames "
                    "after repeated retries (likely disconnected)."
                )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._actual_fps
