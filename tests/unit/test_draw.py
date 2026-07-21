"""Unit tests for webcam_tracker.visualization.draw."""

from __future__ import annotations

import numpy as np

from webcam_tracker.detection import Detection
from webcam_tracker.perf_monitor import PerfSnapshot
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson
from webcam_tracker.visualization import (
    draw_detections,
    draw_perf_overlay,
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
