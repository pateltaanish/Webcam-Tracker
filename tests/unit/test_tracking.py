"""Unit tests for webcam_tracker.tracking.

These use the real ByteTrackTracker (it's fast, deterministic, and pure CPU
-- no model weights or hardware needed), fed synthetic Detection objects.

Important library quirk verified here: a brand-new track is NEVER confirmed
on the frame it's spawned, even with minimum_consecutive_frames=1 -- the
confirmation check only runs when an existing track is re-matched, which
can't happen until at least the following frame. So every track takes at
least 2 total update() calls before it first appears in PersonTracker's
output, regardless of the threshold value (as long as it's >= 1).
"""

from __future__ import annotations

from webcam_tracker.detection import Detection
from webcam_tracker.tracking import PersonTracker


def _box_at(x: float, y: float, size: float = 40.0, confidence: float = 0.9) -> Detection:
    return Detection(x1=x, y1=y, x2=x + size, y2=y + size, confidence=confidence)


def _make_tracker(minimum_consecutive_frames: int = 1) -> PersonTracker:
    return PersonTracker(
        lost_track_buffer=5,
        frame_rate=30.0,
        track_activation_threshold=0.5,
        minimum_consecutive_frames=minimum_consecutive_frames,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.5,
    )


class TestPersonTracker:
    def test_spawn_frame_is_never_confirmed_even_with_threshold_one(self) -> None:
        tracker = _make_tracker(minimum_consecutive_frames=1)
        first = tracker.update([_box_at(10, 10)])
        assert first == []  # not yet confirmed -- must not leak the -1 sentinel

        second = tracker.update([_box_at(11, 11)])
        assert len(second) == 1
        assert second[0].track_id >= 0

    def test_new_track_withheld_until_consecutive_frames_confirm_it(self) -> None:
        tracker = _make_tracker(minimum_consecutive_frames=2)

        first = tracker.update([_box_at(10, 10)])
        assert first == []

        second = tracker.update([_box_at(11, 11)])
        assert len(second) == 1
        assert second[0].track_id >= 0

    def test_track_id_persists_across_frames_for_same_person(self) -> None:
        tracker = _make_tracker(minimum_consecutive_frames=1)
        tracker.update([_box_at(10, 10)])  # spawn frame, unconfirmed
        second = tracker.update([_box_at(12, 12)])
        third = tracker.update([_box_at(14, 14)])
        assert second[0].track_id == third[0].track_id

    def test_no_detections_returns_empty_list(self) -> None:
        tracker = _make_tracker(minimum_consecutive_frames=1)
        tracker.update([_box_at(10, 10)])
        result = tracker.update([])
        assert result == []

    def test_two_simultaneous_people_get_different_ids(self) -> None:
        tracker = _make_tracker(minimum_consecutive_frames=1)
        tracker.update([_box_at(10, 10), _box_at(500, 500)])  # spawn frame
        result = tracker.update([_box_at(11, 11), _box_at(501, 501)])
        assert len(result) == 2
        assert result[0].track_id != result[1].track_id

    def test_reset_clears_known_track_bookkeeping(self) -> None:
        tracker = _make_tracker(minimum_consecutive_frames=1)
        tracker.update([_box_at(10, 10)])
        tracker.update([_box_at(11, 11)])
        assert tracker._known_track_ids != set()  # noqa: SLF001 -- verifying reset's effect

        tracker.reset()

        assert tracker._known_track_ids == set()  # noqa: SLF001
        # Tracker must still work normally after reset.
        tracker.update([_box_at(10, 10)])
        result = tracker.update([_box_at(11, 11)])
        assert len(result) == 1

    def test_tracked_person_center_width_height(self) -> None:
        tracker = _make_tracker(minimum_consecutive_frames=1)
        tracker.update([_box_at(10, 20, size=40)])  # spawn frame
        result = tracker.update([_box_at(10, 20, size=40)])
        person = result[0]
        assert person.center == (30.0, 40.0)
        assert person.width == 40.0
        assert person.height == 40.0
