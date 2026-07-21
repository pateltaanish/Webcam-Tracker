"""Unit tests for webcam_tracker.target_selection."""

from __future__ import annotations

from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson


def _person(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackedPerson:
    return TrackedPerson(track_id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)


class TestSelectAndClear:
    def test_select_sets_target_id(self) -> None:
        selector = TargetSelector()
        selector.select(5)
        assert selector.target_id == 5

    def test_clear_resets_target_id(self) -> None:
        selector = TargetSelector()
        selector.select(5)
        selector.clear()
        assert selector.target_id is None

    def test_clear_when_nothing_selected_is_a_no_op(self) -> None:
        selector = TargetSelector()
        selector.clear()
        assert selector.target_id is None


class TestSelectAtPoint:
    def test_selects_track_containing_point(self) -> None:
        selector = TargetSelector()
        people = [_person(1, 0, 0, 50, 50), _person(2, 100, 100, 150, 150)]
        result = selector.select_at_point((120, 120), people)
        assert result == 2
        assert selector.target_id == 2

    def test_returns_none_and_leaves_selection_unchanged_when_no_box_contains_point(self) -> None:
        selector = TargetSelector()
        selector.select(1)
        people = [_person(1, 0, 0, 50, 50)]
        result = selector.select_at_point((999, 999), people)
        assert result is None
        assert selector.target_id == 1  # unchanged

    def test_picks_smallest_box_when_overlapping(self) -> None:
        selector = TargetSelector()
        big = _person(1, 0, 0, 200, 200)
        small = _person(2, 80, 80, 120, 120)
        result = selector.select_at_point((100, 100), [big, small])
        assert result == 2


class TestStatus:
    def test_none_when_no_target_selected(self) -> None:
        selector = TargetSelector()
        assert selector.status([], frame_width=640, frame_height=480) is None

    def test_visible_false_when_target_not_in_current_frame(self) -> None:
        selector = TargetSelector()
        selector.select(7)
        status = selector.status([], frame_width=640, frame_height=480)
        assert status is not None
        assert status.target_id == 7
        assert status.visible is False
        assert status.tracked_person is None
        assert status.pixel_error is None

    def test_visible_true_with_correct_center_and_pixel_error(self) -> None:
        selector = TargetSelector()
        selector.select(1)
        # Box centered at (100, 100); frame center at (320, 240) for a 640x480 frame.
        people = [_person(1, 80, 80, 120, 120)]
        status = selector.status(people, frame_width=640, frame_height=480)
        assert status is not None
        assert status.visible is True
        assert status.target_center == (100.0, 100.0)
        assert status.frame_center == (320.0, 240.0)
        assert status.pixel_error == (100.0 - 320.0, 100.0 - 240.0)

    def test_normalized_error_matches_pixel_error_divided_by_half_dimensions(self) -> None:
        selector = TargetSelector()
        selector.select(1)
        people = [_person(1, 80, 80, 120, 120)]  # center (100, 100)
        status = selector.status(people, frame_width=640, frame_height=480)
        assert status is not None
        assert status.normalized_error is not None
        expected_dx = (100.0 - 320.0) / 320.0
        expected_dy = (100.0 - 240.0) / 240.0
        assert status.normalized_error == (expected_dx, expected_dy)

    def test_status_finds_correct_target_among_multiple_people(self) -> None:
        selector = TargetSelector()
        selector.select(2)
        people = [_person(1, 0, 0, 10, 10), _person(2, 300, 220, 340, 260)]
        status = selector.status(people, frame_width=640, frame_height=480)
        assert status is not None
        assert status.tracked_person is not None
        assert status.tracked_person.track_id == 2
