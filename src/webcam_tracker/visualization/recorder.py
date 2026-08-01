"""Optional recording of preview frames to a video file for offline review.

Built for Pi trial runs where nobody's watching the live window (or there is
no display) -- record what the camera/pipeline actually did, then copy the
file off the device and watch it afterward on a machine with a display.

Why lazy-initialize the writer instead of sizing it from config: the actual
frame size a camera backend delivers can differ from the requested
width/height (a driver may round to the nearest supported mode), and a
VideoWriter opened at the wrong size silently produces an empty/corrupt file
rather than raising -- sizing off the first real frame avoids that mismatch.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)


class FrameRecorder:
    """Writes frames to an MJPG/.avi file. One instance per recording."""

    def __init__(self, output_path: Path, fps: float) -> None:
        self._output_path = output_path
        # Some sources report 0 for fps (e.g. before the first real frame);
        # a VideoWriter opened at 0 fps produces an unplayable file.
        self._fps = fps if fps > 0 else 30.0
        self._writer: cv2.VideoWriter | None = None

    def write(self, image: np.ndarray) -> None:
        """Append one frame. Opens the underlying file on the first call."""
        if self._writer is None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
            self._writer = cv2.VideoWriter(
                str(self._output_path), fourcc, self._fps, (width, height)
            )
            logger.info(
                "Recording started",
                extra={
                    "output_path": str(self._output_path),
                    "size": f"{width}x{height}",
                    "fps": self._fps,
                },
            )
        self._writer.write(image)

    def close(self) -> None:
        """Flush and release the underlying file. Safe to call multiple times."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Recording saved", extra={"output_path": str(self._output_path)})
