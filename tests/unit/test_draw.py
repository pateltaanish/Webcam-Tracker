"""Unit tests for webcam_tracker.visualization.draw."""

from __future__ import annotations

import numpy as np

from webcam_tracker.detection import Detection
from webcam_tracker.gimbal_control import GimbalCommand
from webcam_tracker.perf_monitor import PerfSnapshot
from webcam_tracker.recovery import RecoveryState, RecoveryStatus
from webcam_tracker.state_machine import SystemStatus, TrackingState
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson
from webcam_tracker.visualization import (
    draw_detections,
    draw_gimbal_widget,
    draw_perf_overlay,
    draw_recovery_overlay,
    draw_state_banner,
    draw_target_overlay,
    draw_tracked_people,
)


def _blank_image() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestDrawDetections:
    def test_mutates_and_returns_same_array(self) -> None:
        image = _blank_image()
        result = draw_detections(image, [])
        assert result is image

    def test_empty_list_leaves_image_unchanged(self) -> None:
        image = _blank_image()
        original = image.copy()
        draw_detections(image, [])
        assert np.array_equal(image, original)

    def test_nonempty_changes_pixels(self) -> None:
        image = _blank_image()
        original = image.copy()
        detection = Detection(x1=10, y1=10, x2=90, y2=90, confidence=0.8)
        draw_detections(image, [detection])
        assert not np.array_equal(image, original)


class TestDrawTrackedPeople:
    def test_mutates_and_returns_same_array(self) -> None:
        image = _blank_image()
        result = draw_tracked_people(image, [])
        assert result is image

    def test_nonempty_changes_pixels(self) -> None:
        image = _blank_image()
        original = image.copy()
        person = TrackedPerson(track_id=3, x1=10, y1=10, x2=90, y2=90, confidence=0.8)
        draw_tracked_people(image, [person])
        assert not np.array_equal(image, original)


class TestDrawPerfOverlay:
    def test_changes_pixels(self) -> None:
        image = _blank_image()
        original = image.copy()
        snapshot = PerfSnapshot(fps=30.0, frame_count=42, total_frame_latency_ms=12.3)
        draw_perf_overlay(image, snapshot)
        assert not np.array_equal(image, original)

    def test_returns_same_array(self) -> None:
        image = _blank_image()
        snapshot = PerfSnapshot(fps=30.0, frame_count=1, total_frame_latency_ms=1.0)
        result = draw_perf_overlay(image, snapshot)
        assert result is image


class TestDrawTargetOverlay:
    def test_none_status_leaves_image_unchanged(self) -> None:
        image = _blank_image()
        original = image.copy()
        draw_target_overlay(image, None)
        assert np.array_equal(image, original)

    def test_lost_target_changes_pixels(self) -> None:
        selector = TargetSelector()
        selector.select(1)
        image = _blank_image()
        original = image.copy()
        status = selector.status([], frame_width=100, frame_height=100)
        draw_target_overlay(image, status)
        assert not np.array_equal(image, original)

    def test_visible_target_changes_pixels(self) -> None:
        selector = TargetSelector()
        selector.select(1)
        person = TrackedPerson(track_id=1, x1=10, y1=10, x2=90, y2=90, confidence=0.9)
        image = _blank_image()
        original = image.copy()
        status = selector.status([person], frame_width=100, frame_height=100)
        draw_target_overlay(image, status)
        assert not np.array_equal(image, original)

    def test_returns_same_array(self) -> None:
        image = _blank_image()
        result = draw_target_overlay(image, None)
        assert result is image


def _command(
    pan_saturated: bool = False,
    tilt_saturated: bool = False,
    emergency_stopped: bool = False,
    target_visible: bool = True,
) -> GimbalCommand:
    return GimbalCommand(
        pan_velocity_deg_s=5.0,
        tilt_velocity_deg_s=-2.0,
        pan_angle_deg=10.0,
        tilt_angle_deg=-5.0,
        pan_saturated=pan_saturated,
        tilt_saturated=tilt_saturated,
        target_visible=target_visible,
        emergency_stopped=emergency_stopped,
    )


class TestDrawGimbalWidget:
    def test_returns_same_array(self) -> None:
        image = _blank_image()
        result = draw_gimbal_widget(
            image,
            _command(),
            pan_limit_deg=170.0,
            tilt_limit_deg=60.0,
            origin=(10, 10),
            size=(40, 40),
        )
        assert result is image

    def test_changes_pixels(self) -> None:
        image = _blank_image()
        original = image.copy()
        draw_gimbal_widget(
            image,
            _command(),
            pan_limit_deg=170.0,
            tilt_limit_deg=60.0,
            origin=(10, 10),
            size=(40, 40),
        )
        assert not np.array_equal(image, original)

    def test_saturated_and_normal_render_differently(self) -> None:
        image_normal = _blank_image()
        draw_gimbal_widget(
            image_normal,
            _command(pan_saturated=False, tilt_saturated=False),
            pan_limit_deg=170.0,
            tilt_limit_deg=60.0,
            origin=(10, 10),
            size=(40, 40),
        )
        image_saturated = _blank_image()
        draw_gimbal_widget(
            image_saturated,
            _command(pan_saturated=True),
            pan_limit_deg=170.0,
            tilt_limit_deg=60.0,
            origin=(10, 10),
            size=(40, 40),
        )
        assert not np.array_equal(image_normal, image_saturated)

    def test_default_origin_does_not_crash_on_larger_image(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        original = image.copy()
        draw_gimbal_widget(image, _command(), pan_limit_deg=170.0, tilt_limit_deg=60.0)
        assert not np.array_equal(image, original)


class TestDrawRecoveryOverlay:
    def test_returns_same_array(self) -> None:
        image = _blank_image()
        result = draw_recovery_overlay(image, RecoveryStatus(state=RecoveryState.IDLE))
        assert result is image

    def test_idle_leaves_image_unchanged(self) -> None:
        image = _blank_image()
        original = image.copy()
        draw_recovery_overlay(image, RecoveryStatus(state=RecoveryState.TRACKING))
        assert np.array_equal(image, original)

    def test_searching_changes_pixels(self) -> None:
        image = _blank_image()
        original = image.copy()
        status = RecoveryStatus(
            state=RecoveryState.SEARCHING,
            search_error=(0.3, 0.0),
            predicted_position=(50.0, 50.0),
        )
        draw_recovery_overlay(image, status)
        assert not np.array_equal(image, original)

    def test_radius_circle_renders_differently(self) -> None:
        status = RecoveryStatus(
            state=RecoveryState.SEARCHING,
            search_error=(0.3, 0.0),
            predicted_position=(50.0, 50.0),
        )
        without_circle = _blank_image()
        draw_recovery_overlay(without_circle, status)
        with_circle = _blank_image()
        draw_recovery_overlay(with_circle, status, reacquire_radius_fraction=0.3)
        assert not np.array_equal(without_circle, with_circle)


def _system_status(
    state: TrackingState = TrackingState.TRACKING,
    needs_drone_assist: bool = False,
) -> SystemStatus:
    return SystemStatus(
        state=state,
        time_in_state_s=1.5,
        target_status=None,
        recovery_status=RecoveryStatus(state=RecoveryState.IDLE),
        gimbal_command=_command(),
        needs_drone_assist=needs_drone_assist,
        reacquired_track_id=None,
    )


def _wide_image() -> np.ndarray:
    # Wide enough that the full banner text (incl. the DRONE-ASSIST suffix)
    # fits without clipping at the image edge.
    return np.zeros((120, 640, 3), dtype=np.uint8)


class TestDrawStateBanner:
    def test_returns_same_array(self) -> None:
        image = _wide_image()
        assert draw_state_banner(image, _system_status()) is image

    def test_changes_pixels(self) -> None:
        image = _wide_image()
        original = image.copy()
        draw_state_banner(image, _system_status())
        assert not np.array_equal(image, original)

    def test_drone_assist_renders_differently(self) -> None:
        without = _wide_image()
        draw_state_banner(without, _system_status(needs_drone_assist=False))
        with_assist = _wide_image()
        draw_state_banner(with_assist, _system_status(needs_drone_assist=True))
        assert not np.array_equal(without, with_assist)
