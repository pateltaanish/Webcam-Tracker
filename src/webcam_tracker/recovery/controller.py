"""Target-loss recovery mechanics (Stage 1.8).

When the selected target's track disappears, this controller:

  1. remembers the target's last-known position and velocity (the latter from
     motion_prediction) while it was still visible;
  2. once it's lost, extrapolates where it probably went and drives a
     simulated side-to-side search sweep (a normalized error fed to the
     gimbal, biased toward the predicted direction);
  3. re-locks onto the nearest *newly-appearing* track that shows up near the
     predicted region, and recommends its track_id back to the caller.

IMPORTANT -- this is IDENTITY-FREE. Step 3 grabs whoever walks into the
predicted spot; it does not check that they're the same person. It is a
deliberate placeholder for Stage 2's face/re-id-gated reacquisition, which
slots in at exactly this decision point. Until then, treat a "reacquired"
result as "something plausible showed up where we expected," not "we
recognized the target."

Scope: this owns only the loss->search->reacquire mechanics. It does NOT own
the overall system state (that's state_machine, Stage 1.9), and it does not
itself change the selection -- it *recommends* a track_id and lets the caller
(TargetSelector) act, keeping selection in one place.

Pure logic -- no OpenCV. Injectable clock (default time.monotonic) for
deterministic tests, matching AxisController / MotionPredictor.
"""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from webcam_tracker.logging_utils import get_logger
from webcam_tracker.motion_prediction import TrackMotion
from webcam_tracker.target_selection import TargetStatus
from webcam_tracker.tracking import TrackedPerson

logger = get_logger(__name__)


class RecoveryState(enum.Enum):
    """Where the recovery controller is in the loss/reacquire cycle."""

    IDLE = "idle"  # no target selected
    TRACKING = "tracking"  # target selected and visible -- nothing to recover
    SEARCHING = "searching"  # target lost, sweeping toward its predicted direction
    REACQUIRED = "reacquired"  # a candidate matched this frame (recommend re-lock)
    GAVE_UP = "gave_up"  # searched past give_up_seconds with no candidate


@dataclass(frozen=True)
class RecoveryStatus:
    """The recovery controller's output for one frame."""

    state: RecoveryState
    # Normalized (-1..1) error to feed the gimbal while SEARCHING, else None.
    search_error: tuple[float, float] | None = None
    # Pixel estimate of where the target is while SEARCHING/REACQUIRED, else None.
    predicted_position: tuple[float, float] | None = None
    # Track to re-lock onto, set only on the REACQUIRED frame. The caller is
    # expected to pass this to TargetSelector.select().
    reacquire_track_id: int | None = None


class RecoveryController:
    def __init__(
        self,
        give_up_seconds: float,
        prediction_horizon_seconds: float,
        reacquire_radius_fraction: float,
        search_amplitude: float,
        search_period_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._give_up_seconds = give_up_seconds
        self._prediction_horizon_seconds = prediction_horizon_seconds
        self._reacquire_radius_fraction = reacquire_radius_fraction
        self._search_amplitude = search_amplitude
        self._search_period_seconds = search_period_seconds
        self._clock = clock

        self._state = RecoveryState.IDLE
        self._last_position: tuple[float, float] | None = None
        self._last_velocity: tuple[float, float] = (0.0, 0.0)
        self._ids_at_loss: set[int] = set()
        self._search_started_at: float | None = None

    @property
    def state(self) -> RecoveryState:
        return self._state

    def update(
        self,
        target_status: TargetStatus | None,
        tracked_people: Sequence[TrackedPerson],
        target_motion: TrackMotion | None,
        frame_width: int,
        frame_height: int,
    ) -> RecoveryStatus:
        """Advance the recovery state machine by one frame.

        `target_status` is TargetSelector's output (None if nothing selected);
        `target_motion` is motion_prediction's estimate for the selected
        target (None if unavailable, e.g. filter not warmed up yet).
        """
        if target_status is None:
            self._transition(RecoveryState.IDLE)
            self._last_position = None
            return RecoveryStatus(state=RecoveryState.IDLE)

        if target_status.visible:
            self._record_visible(target_status, tracked_people, target_motion)
            self._transition(RecoveryState.TRACKING)
            return RecoveryStatus(state=RecoveryState.TRACKING)

        # Target selected but not in this frame -> lost.
        return self._search(tracked_people, frame_width, frame_height)

    def _record_visible(
        self,
        target_status: TargetStatus,
        tracked_people: Sequence[TrackedPerson],
        target_motion: TrackMotion | None,
    ) -> None:
        """Refresh last-known state each frame the target is visible, so the
        moment it vanishes we already have a good snapshot to search from."""
        if target_status.target_center is not None:
            self._last_position = target_status.target_center
        if target_motion is not None:
            self._last_velocity = target_motion.velocity
        # Remember who was on screen now, so once the target is lost we only
        # consider tracks that appear AFTER this as reacquisition candidates
        # (a bystander who was already here isn't "the target reappearing").
        self._ids_at_loss = {p.track_id for p in tracked_people}
        self._search_started_at = None

    def _search(
        self, tracked_people: Sequence[TrackedPerson], frame_width: int, frame_height: int
    ) -> RecoveryStatus:
        now = self._clock()
        if self._state not in (RecoveryState.SEARCHING, RecoveryState.GAVE_UP):
            # Just became lost this frame -> start the search clock.
            self._search_started_at = now
            self._transition(RecoveryState.SEARCHING)

        if self._state == RecoveryState.GAVE_UP:
            return RecoveryStatus(state=RecoveryState.GAVE_UP)

        assert self._search_started_at is not None
        elapsed = max(now - self._search_started_at, 0.0)
        if elapsed > self._give_up_seconds:
            self._transition(RecoveryState.GAVE_UP)
            return RecoveryStatus(state=RecoveryState.GAVE_UP)

        predicted = self._predicted_position(elapsed, frame_width, frame_height)

        candidate = self._reacquire_candidate(tracked_people, predicted, frame_width, frame_height)
        if candidate is not None:
            logger.info(
                "Recovery reacquired a candidate (identity-free placeholder)",
                extra={"track_id": candidate.track_id, "elapsed_s": round(elapsed, 2)},
            )
            self._transition(RecoveryState.REACQUIRED)
            return RecoveryStatus(
                state=RecoveryState.REACQUIRED,
                reacquire_track_id=candidate.track_id,
                predicted_position=predicted,
            )

        search_error = self._search_error(elapsed, predicted, frame_width, frame_height)
        return RecoveryStatus(
            state=RecoveryState.SEARCHING,
            search_error=search_error,
            predicted_position=predicted,
        )

    def _predicted_position(
        self, elapsed: float, frame_width: int, frame_height: int
    ) -> tuple[float, float]:
        """Where the target probably is now: last-known position plus its
        last-known velocity extrapolated forward, capped at the prediction
        horizon and clamped to stay on screen."""
        if self._last_position is None:
            return (frame_width / 2.0, frame_height / 2.0)
        horizon = min(elapsed, self._prediction_horizon_seconds)
        px = self._last_position[0] + self._last_velocity[0] * horizon
        py = self._last_position[1] + self._last_velocity[1] * horizon
        px = _clamp(px, 0.0, float(frame_width))
        py = _clamp(py, 0.0, float(frame_height))
        return px, py

    def _reacquire_candidate(
        self,
        tracked_people: Sequence[TrackedPerson],
        predicted: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> TrackedPerson | None:
        """Nearest newly-appeared track within the reacquire radius of the
        predicted position, or None. 'Newly-appeared' = a track_id that
        wasn't on screen when we lost the target."""
        radius = self._reacquire_radius_fraction * math.hypot(frame_width, frame_height)
        best: TrackedPerson | None = None
        best_distance = radius
        for person in tracked_people:
            if person.track_id in self._ids_at_loss:
                continue
            distance = math.hypot(person.center[0] - predicted[0], person.center[1] - predicted[1])
            if distance <= best_distance:
                best = person
                best_distance = distance
        return best

    def _search_error(
        self,
        elapsed: float,
        predicted: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        """Normalized (-1..1) error fed to the gimbal while searching: point
        toward the predicted position, with a horizontal sweep added so the
        (simulated) camera scans around it instead of freezing."""
        cx, cy = frame_width / 2.0, frame_height / 2.0
        base_pan = _clamp((predicted[0] - cx) / (frame_width / 2.0), -1.0, 1.0)
        base_tilt = _clamp((predicted[1] - cy) / (frame_height / 2.0), -1.0, 1.0)
        phase = 2.0 * math.pi * elapsed / self._search_period_seconds
        pan = _clamp(base_pan + self._search_amplitude * math.sin(phase), -1.0, 1.0)
        return pan, base_tilt

    def _transition(self, new_state: RecoveryState) -> None:
        if new_state == self._state:
            return
        logger.info(
            "Recovery state change", extra={"from": self._state.value, "to": new_state.value}
        )
        self._state = new_state


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
