"""Drawing helpers shared by every preview/debug script.

Every function here mutates and returns the `image` it's given (the same
convention OpenCV's own `cv2.rectangle`/`cv2.putText` use), rather than
returning a copy -- this lets a caller chain several draw calls onto one
frame without redundant copies. Copy the source frame first if you need to
keep the original untouched, e.g. ``annotated = frame.image.copy()``.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from webcam_tracker.detection import Detection
from webcam_tracker.perf_monitor import PerfSnapshot
from webcam_tracker.target_selection import TargetStatus
from webcam_tracker.tracking import TrackedPerson
from webcam_tracker.visualization.colors import assign_colors, color_for_track_id

_TARGET_HIGHLIGHT_COLOR = (
    0,
    255,
    255,
)  # fixed bright yellow -- stands out regardless of track_id color
_LOST_TEXT_COLOR = (0, 0, 255)  # red

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draws a box + confidence per raw detection, colored by screen position
    (see assign_colors -- detections have no persistent identity to color by)."""
    colors = assign_colors(detections)
    for detection, color in zip(detections, colors, strict=True):
        x1, y1, x2, y2 = (int(v) for v in (detection.x1, detection.y1, detection.x2, detection.y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"person {detection.confidence:.2f}"
        cv2.putText(image, label, (x1, max(0, y1 - 8)), _FONT, 0.6, color, 2)
    return image


def draw_tracked_people(image: np.ndarray, tracked_people: Sequence[TrackedPerson]) -> np.ndarray:
    """Draws a box + track ID + confidence per tracked person, colored by
    track_id -- the same person keeps the same color across frames."""
    for person in tracked_people:
        x1, y1, x2, y2 = (int(v) for v in (person.x1, person.y1, person.x2, person.y2))
        color = color_for_track_id(person.track_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"ID {person.track_id} {person.confidence:.2f}"
        cv2.putText(image, label, (x1, max(0, y1 - 8)), _FONT, 0.6, color, 2)
    return image


def draw_perf_overlay(
    image: np.ndarray,
    snapshot: PerfSnapshot,
    extra_text: str = "",
    origin: tuple[int, int] = (10, 30),
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draws an FPS/frame-count line (plus optional caller-supplied extra
    text, e.g. "tracked: 2") in the corner of the frame."""
    text = f"FPS: {snapshot.fps:.1f}  frame: {snapshot.frame_count}"
    if extra_text:
        text += f"  {extra_text}"
    cv2.putText(image, text, origin, _FONT, 0.8, color, 2)
    return image


def draw_target_overlay(image: np.ndarray, status: TargetStatus | None) -> np.ndarray:
    """Highlights the currently selected target: a crosshair at frame center,
    a thick fixed-color box + crosshair on the target, a line between them
    (the visual form of the pixel error), and an error readout. If the
    target isn't visible this frame, draws a "TARGET LOST" warning instead.
    No-ops if no target is selected at all (status is None).
    """
    if status is None:
        return image

    height, width = image.shape[:2]
    frame_cx, frame_cy = (int(v) for v in status.frame_center)
    cv2.drawMarker(
        image, (frame_cx, frame_cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20
    )

    if not status.visible:
        cv2.putText(
            image,
            f"TARGET LOST (id={status.target_id})",
            (10, height - 20),
            _FONT,
            0.8,
            _LOST_TEXT_COLOR,
            2,
        )
        return image

    assert status.tracked_person is not None
    assert status.target_center is not None
    assert status.pixel_error is not None
    assert status.normalized_error is not None

    person = status.tracked_person
    x1, y1, x2, y2 = (int(v) for v in (person.x1, person.y1, person.x2, person.y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), _TARGET_HIGHLIGHT_COLOR, 3)

    target_cx, target_cy = (int(v) for v in status.target_center)
    cv2.drawMarker(
        image, (target_cx, target_cy), _TARGET_HIGHLIGHT_COLOR, markerType=cv2.MARKER_CROSS
    )
    cv2.line(image, (frame_cx, frame_cy), (target_cx, target_cy), _TARGET_HIGHLIGHT_COLOR, 1)

    dx, dy = status.pixel_error
    norm_dx, norm_dy = status.normalized_error
    error_text = (
        f"target id={status.target_id} err=({dx:+.0f},{dy:+.0f})px "
        f"norm=({norm_dx:+.2f},{norm_dy:+.2f})"
    )
    cv2.putText(image, error_text, (10, height - 20), _FONT, 0.6, _TARGET_HIGHLIGHT_COLOR, 2)
    return image
