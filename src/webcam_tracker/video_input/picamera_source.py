"""Live capture from a Raspberry Pi CSI camera (e.g. the AI Camera / IMX500)
via picamera2/libcamera.

Why a separate source instead of teaching WebcamSource a new backend: CSI
cameras aren't UVC devices -- there's no OpenCV VideoCapture index to open at
all, the whole capture API is different (picamera2, not cv2). Downstream code
never notices, because both classes produce the same FrameSource contract.

``picamera2`` only exists on Raspberry Pi OS (it wraps libcamera bindings
installed via ``apt install python3-picamera2``, not a portable pip wheel),
so it's imported lazily inside open() rather than at module load time --
importing this module on a dev machine without picamera2 installed must not
break anything that merely imports the video_input package.

Color order gotcha: picamera2's format name "RGB888" is, confusingly, the
memory byte order -- for this camera stack it comes out BGR when read into a
numpy array, which happens to match what VideoFrame.image promises (BGR,
matching OpenCV's native layout) with no extra conversion. This is a known
picamera2 quirk, not a typo. Verify on real hardware before trusting it
(point the camera at something solid red and check image[y, x] against
image[y, x, ::-1]) -- a silent channel swap here produces a pipeline that
runs fine but feeds wrong colors into face/Re-ID.
"""

from __future__ import annotations

import time

import numpy as np

from webcam_tracker.logging_utils import get_logger
from webcam_tracker.video_input.base import FrameSource, VideoFrame, VideoSourceError

logger = get_logger(__name__)


class PiCameraSource(FrameSource):
    """Captures frames from a CSI camera via picamera2."""

    def __init__(self, requested_width: int, requested_height: int, requested_fps: int) -> None:
        self._requested_width = requested_width
        self._requested_height = requested_height
        self._requested_fps = requested_fps
        self._picam2: object | None = None
        self._actual_fps = 0.0
        self._frame_index = 0

    def open(self) -> None:
        if self._picam2 is not None:
            logger.warning("open() called while already open; closing previous capture first")
            self.close()

        try:
            from picamera2 import Picamera2  # noqa: PLC0415 -- Pi-only, see module docstring
        except ImportError as exc:
            raise VideoSourceError(
                "picamera2 is not installed. It's only available on Raspberry Pi OS via "
                "`sudo apt install -y python3-picamera2` (not pip) -- see "
                "docs/03_onboard_computer.md."
            ) from exc

        picam2 = Picamera2()
        frame_duration_us = int(1_000_000 / self._requested_fps)
        config = picam2.create_video_configuration(
            main={
                "size": (self._requested_width, self._requested_height),
                "format": "RGB888",  # see module docstring: yields BGR-ordered arrays
            },
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        )
        try:
            picam2.configure(config)
            picam2.start()
        except Exception as exc:
            picam2.close()
            raise VideoSourceError(f"Could not start Pi camera: {exc}") from exc

        self._picam2 = picam2
        # picamera2 has no reported-actual-fps readback comparable to
        # CAP_PROP_FPS; fall back to the requested rate, same as WebcamSource
        # does when a driver reports 0.
        self._actual_fps = float(self._requested_fps)

        logger.info(
            "Pi camera opened",
            extra={
                "requested_resolution": f"{self._requested_width}x{self._requested_height}",
                "requested_fps": self._requested_fps,
            },
        )

    def read(self) -> VideoFrame | None:
        if self._picam2 is None:
            raise VideoSourceError("read() called before open()")

        try:
            image = self._picam2.capture_array("main")  # type: ignore[attr-defined]
        except Exception as exc:
            raise VideoSourceError(f"Pi camera stopped producing frames: {exc}") from exc

        frame = VideoFrame(
            image=np.ascontiguousarray(image[:, :, :3]),
            frame_index=self._frame_index,
            timestamp=time.monotonic(),
            source_fps=self._actual_fps,
        )
        self._frame_index += 1
        return frame

    def close(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()  # type: ignore[attr-defined]
            self._picam2.close()  # type: ignore[attr-defined]
            self._picam2 = None

    @property
    def fps(self) -> float:
        return self._actual_fps
