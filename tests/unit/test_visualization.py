"""Unit tests for webcam_tracker.visualization.colors."""

from __future__ import annotations

from webcam_tracker.detection import Detection
from webcam_tracker.visualization import PALETTE, assign_colors, color_for_track_id


def _detection_at_x(center_x: float) -> Detection:
    # Only x1/x2 matter for center(); y/confidence are arbitrary.
    half_width = 10.0
    return Detection(
        x1=center_x - half_width, y1=0.0, x2=center_x + half_width, y2=50.0, confidence=0.9
    )


def test_empty_input_returns_empty_list() -> None:
    assert assign_colors([]) == []


def test_single_detection_gets_first_palette_color() -> None:
    colors = assign_colors([_detection_at_x(100.0)])
    assert colors == [PALETTE[0]]


def test_colors_assigned_by_left_to_right_position_not_input_order() -> None:
    # Rightmost detection listed first -- output order must still match
    # input order, but color rank must follow position (leftmost = color 0).
    rightmost = _detection_at_x(300.0)
    leftmost = _detection_at_x(50.0)
    middle = _detection_at_x(150.0)

    colors = assign_colors([rightmost, leftmost, middle])

    assert colors[1] == PALETTE[0]  # leftmost
    assert colors[2] == PALETTE[1]  # middle
    assert colors[0] == PALETTE[2]  # rightmost


def test_more_detections_than_palette_colors_wraps_around() -> None:
    detections = [_detection_at_x(float(x)) for x in range(len(PALETTE) + 2)]
    colors = assign_colors(detections)
    assert colors[0] == PALETTE[0]
    assert colors[len(PALETTE)] == PALETTE[0]  # wrapped
    assert colors[len(PALETTE) + 1] == PALETTE[1]


def test_color_for_track_id_is_deterministic() -> None:
    assert color_for_track_id(7) == color_for_track_id(7)


def test_color_for_track_id_differs_for_different_ids_within_palette_size() -> None:
    assert color_for_track_id(0) != color_for_track_id(1)


def test_color_for_track_id_wraps_around_palette() -> None:
    assert color_for_track_id(0) == color_for_track_id(len(PALETTE))
