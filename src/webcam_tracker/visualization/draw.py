"""Drawing helpers shared by every preview/debug script.

Every function here mutates and returns the `image` it's given (the same
convention OpenCV's own `cv2.rectangle`/`cv2.putText` use), rather than
returning a copy -- this lets a caller chain several draw calls onto one
frame without redundant copies. Copy the source frame first if you need to
keep the original untouched, e.g. ``annotated = frame.image.copy()``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from webcam_tracker.detection import Detection
from webcam_tracker.gimbal_control import GimbalCommand
from webcam_tracker.perf_monitor import PerfSnapshot
from webcam_tracker.recovery import RecoveryState, RecoveryStatus
from webcam_tracker.state_machine import SystemStatus, TrackingState
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


def draw_gimbal_widget(
    image: np.ndarray,
    command: GimbalCommand,
    pan_limit_deg: float,
    tilt_limit_deg: float,
    origin: tuple[int, int] | None = None,
    size: tuple[int, int] = (160, 160),
) -> np.ndarray:
    """Draws a small attitude-indicator-style widget: a bordered box
    representing the gimbal's full pan/tilt range, with a dot showing its
    current simulated position (green = normal, red = an axis is saturated,
    orange = emergency stopped), plus a numeric readout below it. Defaults
    to the top-right corner if `origin` isn't given.
    """
    if origin is None:
        origin = (image.shape[1] - size[0] - 10, 10)
    x0, y0 = origin
    width, height = size

    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (200, 200, 200), 1)
    center_x, center_y = x0 + width // 2, y0 + height // 2
    cv2.drawMarker(
        image, (center_x, center_y), (150, 150, 150), markerType=cv2.MARKER_CROSS, markerSize=8
    )

    pan_fraction = _fraction_within_limit(command.pan_angle_deg, pan_limit_deg)
    tilt_fraction = _fraction_within_limit(command.tilt_angle_deg, tilt_limit_deg)
    dot_x = int(x0 + pan_fraction * width)
    # Positive tilt = "up" -> screen y decreases as tilt increases, so invert.
    dot_y = int(y0 + (1.0 - tilt_fraction) * height)

    if command.emergency_stopped:
        dot_color = (0, 165, 255)  # orange
    elif command.pan_saturated or command.tilt_saturated:
        dot_color = (0, 0, 255)  # red
    else:
        dot_color = (0, 255, 0)  # green
    cv2.circle(image, (dot_x, dot_y), 6, dot_color, -1)

    pan_flag = " SAT" if command.pan_saturated else ""
    tilt_flag = " SAT" if command.tilt_saturated else ""
    lines = [
        f"pan {command.pan_angle_deg:+.1f}deg v={command.pan_velocity_deg_s:+.1f}{pan_flag}",
        f"tilt {command.tilt_angle_deg:+.1f}deg v={command.tilt_velocity_deg_s:+.1f}{tilt_flag}",
    ]
    if command.emergency_stopped:
        lines.append("E-STOP")
    elif not command.target_visible:
        lines.append("no target")

    for i, line in enumerate(lines):
        cv2.putText(image, line, (x0, y0 + height + 18 + i * 16), _FONT, 0.45, (255, 255, 255), 1)
    return image


_TRACKING_STATE_COLORS: dict[TrackingState, tuple[int, int, int]] = {
    TrackingState.IDLE: (180, 180, 180),  # gray
    TrackingState.TRACKING: (0, 255, 0),  # green
    TrackingState.TEMPORARILY_OCCLUDED: (0, 255, 255),  # yellow
    TrackingState.RECOVERY_SEARCH: (0, 165, 255),  # orange
    TrackingState.SAFE_HOVER_REQUESTED: (0, 0, 255),  # red
    TrackingState.STOPPED: (0, 0, 255),  # red
}


def draw_state_banner(
    image: np.ndarray, status: SystemStatus, origin: tuple[int, int] = (10, 90)
) -> np.ndarray:
    """Draws the authoritative system state (from the state machine) as a
    prominent labeled banner: the state name, how long it's been in that
    state, and a DRONE-ASSIST flag when the gimbal is saturated and the drone
    body would need to help. Colored by state (green tracking -> yellow
    occluded -> orange searching -> red hover/stop)."""
    color = _TRACKING_STATE_COLORS.get(status.state, (255, 255, 255))
    text = f"[{status.state.value.upper()}] {status.time_in_state_s:.1f}s"
    if status.needs_drone_assist:
        text += "  DRONE-ASSIST"
    cv2.putText(image, text, origin, _FONT, 0.7, color, 2)
    return image


_RECOVERY_STATE_COLORS: dict[RecoveryState, tuple[int, int, int]] = {
    RecoveryState.SEARCHING: (0, 165, 255),  # orange
    RecoveryState.REACQUIRED: (0, 255, 0),  # green
    RecoveryState.GAVE_UP: (0, 0, 255),  # red
}


def draw_recovery_overlay(
    image: np.ndarray,
    status: RecoveryStatus,
    reacquire_radius_fraction: float | None = None,
) -> np.ndarray:
    """Visualizes target-loss recovery: a state banner plus, while searching,
    a marker at the predicted target position and (if `reacquire_radius_fraction`
    is given) the circle within which a newly-appearing track would re-lock.
    No-ops while the target is present (IDLE/TRACKING) -- there's nothing to
    recover, so nothing is drawn.
    """
    if status.state in (RecoveryState.IDLE, RecoveryState.TRACKING):
        return image

    height, width = image.shape[:2]
    color = _RECOVERY_STATE_COLORS.get(status.state, (255, 255, 255))
    cv2.putText(image, f"RECOVERY: {status.state.value.upper()}", (10, 60), _FONT, 0.8, color, 2)

    if status.predicted_position is not None:
        px, py = (int(v) for v in status.predicted_position)
        cv2.drawMarker(
            image, (px, py), color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=24, thickness=2
        )
        cv2.putText(image, "predicted", (px + 14, py + 4), _FONT, 0.5, color, 1)
        if reacquire_radius_fraction is not None:
            radius = int(reacquire_radius_fraction * math.hypot(width, height))
            cv2.circle(image, (px, py), radius, color, 1)
    return image


def _fraction_within_limit(angle_deg: float, limit_deg: float) -> float:
    """Maps angle_deg from [-limit_deg, +limit_deg] to [0.0, 1.0], clamped."""
    if limit_deg <= 0:
        return 0.5
    fraction = (angle_deg / limit_deg + 1.0) / 2.0
    return max(0.0, min(1.0, fraction))
