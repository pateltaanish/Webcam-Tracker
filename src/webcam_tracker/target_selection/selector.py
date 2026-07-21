"""Manual target selection (Stage 1.6).

Selects which track_id from `tracking` is "the target" the rest of the
pipeline should care about. In Stage 1, selection is manual (click a person
on screen); Stage 2+ replaces/extends this with DB-driven identity
selection, but the *interface* -- "here's the current target's status" --
stays the same so downstream modules (gimbal_control, state_machine) don't
need to change when that happens.

This module has no drawing/UI code of its own (see
webcam_tracker.visualization.draw_target_overlay for that) and no OpenCV
dependency -- it's pure selection logic, testable without a camera or a window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from webcam_tracker.logging_utils import get_logger
from webcam_tracker.tracking import TrackedPerson

logger = get_logger(__name__)


@dataclass(frozen=True)
class TargetStatus:
    """The selected target's state for one frame.

    `visible=False` means the target's track_id didn't appear in this
    frame's tracked people -- it does NOT necessarily mean the person is
    gone for good (ByteTrack may still be coasting it within
    tracking.lost_track_buffer); that distinction, and what to do about it,
    belongs to the recovery module (Stage 1.8), not here.
    """

    target_id: int
    visible: bool
    frame_center: tuple[float, float]
    tracked_person: TrackedPerson | None = None
    target_center: tuple[float, float] | None = None
    pixel_error: tuple[float, float] | None = None
    normalized_error: tuple[float, float] | None = None


class TargetSelector:
    """Holds the currently selected target track_id and reports its per-frame status."""

    def __init__(self) -> None:
        self._target_id: int | None = None

    @property
    def target_id(self) -> int | None:
        return self._target_id

    def select(self, track_id: int) -> None:
        if track_id != self._target_id:
            logger.info(
                "Target selected", extra={"track_id": track_id, "previous": self._target_id}
            )
        self._target_id = track_id

    def clear(self) -> None:
        if self._target_id is not None:
            logger.info("Target cleared", extra={"previous_track_id": self._target_id})
        self._target_id = None

    def select_at_point(
        self, point: tuple[float, float], tracked_people: Sequence[TrackedPerson]
    ) -> int | None:
        """Select whichever tracked person's box contains `point`.

        If multiple boxes contain the point (people overlapping on screen),
        picks the smallest box -- typically the nearer/frontmost person.
        Returns the selected track_id, or None (selection left unchanged) if
        no box contains the point.
        """
        x, y = point
        candidates = [p for p in tracked_people if p.x1 <= x <= p.x2 and p.y1 <= y <= p.y2]
        if not candidates:
            return None
        chosen = min(candidates, key=lambda p: p.width * p.height)
        self.select(chosen.track_id)
        return chosen.track_id

    def status(
        self, tracked_people: Sequence[TrackedPerson], frame_width: int, frame_height: int
    ) -> TargetStatus | None:
        """The current target's status against this frame's tracked people.

        Returns None if no target is selected at all (distinct from
        `visible=False`, which means a target IS selected but isn't in this frame).
        """
        if self._target_id is None:
            return None

        frame_center = (frame_width / 2.0, frame_height / 2.0)
        match = next((p for p in tracked_people if p.track_id == self._target_id), None)
        if match is None:
            return TargetStatus(target_id=self._target_id, visible=False, frame_center=frame_center)

        target_center = match.center
        pixel_error = (target_center[0] - frame_center[0], target_center[1] - frame_center[1])
        normalized_error = (
            pixel_error[0] / (frame_width / 2.0),
            pixel_error[1] / (frame_height / 2.0),
        )
        return TargetStatus(
            target_id=self._target_id,
            visible=True,
            frame_center=frame_center,
            tracked_person=match,
            target_center=target_center,
            pixel_error=pixel_error,
            normalized_error=normalized_error,
        )
