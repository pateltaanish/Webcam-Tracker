"""The authoritative tracking state machine (Stage 1.9).

This is the "brain" that ties the pipeline together. Perception (detection +
tracking) runs upstream and hands it a list of tracked people each frame; the
state machine then coordinates the *decision* modules -- target selection,
motion prediction, recovery, and the gimbal -- into ONE state and ONE set of
outputs per frame, and logs every state transition as a structured event
(the audit trail called for in docs/02_architecture.md sec 4).

Why centralize this: before this module existed, the "what should the system
do right now" logic lived as ad-hoc if/elif chains copy-pasted into each
preview script. Pulling it here means the behavior is defined once, tested
once, and auditable as a clean sequence of state changes -- and it's the
single place a future drone_control reads from (see `needs_drone_assist`).

## States (Stage 1 subset of docs/02_architecture.md sec 4)

Implemented now (identity-free, manual selection):
  IDLE, TRACKING, TEMPORARILY_OCCLUDED, RECOVERY_SEARCH,
  SAFE_HOVER_REQUESTED, STOPPED.

Deferred, with their slot-in points noted:
  REGISTERING / CANDIDATE_DETECTED / VERIFYING_IDENTITY -- Stage 2 (identity):
    these gate "is this actually the registered person?" before committing to
    a track. Today TRACKING is entered on manual selection and reacquisition
    is identity-free; Stage 2 inserts verification at the reacquire decision.
  ERROR -- a later hardening pass (unhandled module fault -> suppress outputs).

The doc's TARGET_LOST ("save snapshot, then -> RECOVERY_SEARCH") is folded in:
the recovery controller already snapshots last-known state, so loss goes
TEMPORARILY_OCCLUDED -> RECOVERY_SEARCH directly.

Pure coordination logic -- no OpenCV, no camera. Injectable clock for
deterministic tests, matching the rest of the pipeline.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from webcam_tracker.gimbal_control import GimbalCommand, GimbalController
from webcam_tracker.logging_utils import get_logger
from webcam_tracker.motion_prediction import MotionPredictor
from webcam_tracker.recovery import RecoveryController, RecoveryState, RecoveryStatus
from webcam_tracker.target_selection import TargetSelector, TargetStatus
from webcam_tracker.tracking import TrackedPerson

logger = get_logger(__name__)


class TrackingState(enum.Enum):
    """The system's authoritative mode for one frame (Stage 1 subset)."""

    IDLE = "idle"  # no target selected -> no gimbal/drone output
    TRACKING = "tracking"  # target selected and visible -> follow it
    TEMPORARILY_OCCLUDED = "temporarily_occluded"  # brief dropout -> hold, wait
    RECOVERY_SEARCH = "recovery_search"  # lost too long -> search sweep
    SAFE_HOVER_REQUESTED = "safe_hover_requested"  # search gave up -> hold, alert
    STOPPED = "stopped"  # emergency stop engaged -> all output suppressed


# A recovery status meaning "recovery is not engaged this frame" -- used when
# the state machine holds (occlusion / idle / stopped) rather than searching.
_RECOVERY_NOT_ENGAGED = RecoveryStatus(state=RecoveryState.IDLE)


@dataclass(frozen=True)
class SystemStatus:
    """Everything the state machine decided for one frame."""

    state: TrackingState
    time_in_state_s: float
    target_status: TargetStatus | None
    recovery_status: RecoveryStatus
    gimbal_command: GimbalCommand
    # True when the gimbal is at a limit while it should be following the
    # target (TRACKING/RECOVERY_SEARCH) -- i.e. the gimbal alone can't keep the
    # target centered and the drone body would need to move. This is the
    # concrete signal the (future) drone_control module consumes; nothing acts
    # on it yet.
    needs_drone_assist: bool
    # Set on the single frame a re-lock is committed (identity-free), else None.
    reacquired_track_id: int | None


class TrackingStateMachine:
    """Coordinates selection + prediction + recovery + gimbal each frame.

    Owns those four decision components (perception -- detection/tracking -- is
    upstream and feeds `update`). The selector is exposed so a UI can drive
    manual selection (clicks); the state machine drives it during recovery.
    """

    def __init__(
        self,
        selector: TargetSelector,
        predictor: MotionPredictor,
        recovery: RecoveryController,
        gimbal: GimbalController,
        occlusion_timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._selector = selector
        self._predictor = predictor
        self._recovery = recovery
        self._gimbal = gimbal
        self._occlusion_timeout = occlusion_timeout_seconds
        self._clock = clock

        self._state = TrackingState.IDLE
        self._state_since = clock()
        self._lost_since: float | None = None

    @property
    def selector(self) -> TargetSelector:
        """The target selector, for wiring UI click-to-select."""
        return self._selector

    @property
    def state(self) -> TrackingState:
        return self._state

    def emergency_stop(self) -> None:
        """System-wide stop: freezes the gimbal until resume(). While engaged,
        update() reports STOPPED and suppresses all control output."""
        self._gimbal.emergency_stop()

    def resume(self) -> None:
        self._gimbal.resume()

    def update(
        self, tracked_people: Sequence[TrackedPerson], frame_width: int, frame_height: int
    ) -> SystemStatus:
        """Advance the whole control pipeline by one frame and return the
        coordinated decision. `tracked_people` is this frame's confirmed
        tracks from the (upstream) detector + tracker."""
        now = self._clock()
        self._predictor.update(tracked_people)

        target_id = self._selector.target_id
        target_status = self._selector.status(tracked_people, frame_width, frame_height)
        visible = target_status is not None and target_status.visible
        target_motion = self._predictor.motion(target_id) if target_id is not None else None

        # Maintain the "how long has the target been missing" timer.
        if target_status is None or visible:
            self._lost_since = None
        elif self._lost_since is None:
            self._lost_since = now

        # Emergency stop overrides every other consideration.
        if self._gimbal.is_emergency_stopped:
            if visible:
                # Keep recovery's last-known snapshot warm so a resume mid-track
                # doesn't lose where the target was.
                self._recovery.update(
                    target_status, tracked_people, target_motion, frame_width, frame_height
                )
            command = self._gimbal.compute(None, None)
            return self._finish(
                TrackingState.STOPPED,
                now,
                target_status,
                _RECOVERY_NOT_ENGAGED,
                command,
                False,
                None,
            )

        # No target selected at all.
        if target_status is None:
            recovery_status = self._recovery.update(
                None, tracked_people, None, frame_width, frame_height
            )
            command = self._gimbal.compute(None, None)
            return self._finish(
                TrackingState.IDLE, now, None, recovery_status, command, False, None
            )

        # Target selected and visible -> follow it.
        if visible:
            recovery_status = self._recovery.update(
                target_status, tracked_people, target_motion, frame_width, frame_height
            )
            assert target_status.normalized_error is not None
            pan_error, tilt_error = target_status.normalized_error
            command = self._gimbal.compute(pan_error, tilt_error)
            needs = command.pan_saturated or command.tilt_saturated
            return self._finish(
                TrackingState.TRACKING, now, target_status, recovery_status, command, needs, None
            )

        # Target selected but missing this frame.
        assert self._lost_since is not None
        occluded_for = now - self._lost_since
        if occluded_for <= self._occlusion_timeout:
            # Brief dropout: hold and wait for the SAME track id to return (the
            # tracker's lost_track_buffer usually gives it back). Don't engage
            # the search or grab a different person yet.
            command = self._gimbal.compute(None, None)
            return self._finish(
                TrackingState.TEMPORARILY_OCCLUDED,
                now,
                target_status,
                _RECOVERY_NOT_ENGAGED,
                command,
                False,
                None,
            )

        # Occlusion outlasted the grace period -> engage the recovery search.
        recovery_status = self._recovery.update(
            target_status, tracked_people, target_motion, frame_width, frame_height
        )

        if recovery_status.state == RecoveryState.GAVE_UP:
            command = self._gimbal.compute(None, None)
            return self._finish(
                TrackingState.SAFE_HOVER_REQUESTED,
                now,
                target_status,
                recovery_status,
                command,
                False,
                None,
            )

        if recovery_status.reacquire_track_id is not None:
            # Identity-FREE re-lock -- Stage 2 inserts identity verification
            # here (a VERIFYING_IDENTITY state) before committing.
            self._selector.select(recovery_status.reacquire_track_id)
            self._lost_since = None
            relocked = self._selector.status(tracked_people, frame_width, frame_height)
            if relocked is not None and relocked.visible:
                assert relocked.normalized_error is not None
                pan_error, tilt_error = relocked.normalized_error
                command = self._gimbal.compute(pan_error, tilt_error)
                needs = command.pan_saturated or command.tilt_saturated
                return self._finish(
                    TrackingState.TRACKING,
                    now,
                    relocked,
                    recovery_status,
                    command,
                    needs,
                    recovery_status.reacquire_track_id,
                )

        # Actively searching: drive the gimbal with recovery's search sweep.
        search_pan: float | None
        search_tilt: float | None
        if recovery_status.search_error is not None:
            search_pan, search_tilt = recovery_status.search_error
        else:
            search_pan, search_tilt = None, None
        command = self._gimbal.compute(search_pan, search_tilt)
        needs = command.pan_saturated or command.tilt_saturated
        return self._finish(
            TrackingState.RECOVERY_SEARCH, now, target_status, recovery_status, command, needs, None
        )

    def _finish(
        self,
        state: TrackingState,
        now: float,
        target_status: TargetStatus | None,
        recovery_status: RecoveryStatus,
        gimbal_command: GimbalCommand,
        needs_drone_assist: bool,
        reacquired_track_id: int | None,
    ) -> SystemStatus:
        self._transition(state, now)
        return SystemStatus(
            state=state,
            time_in_state_s=now - self._state_since,
            target_status=target_status,
            recovery_status=recovery_status,
            gimbal_command=gimbal_command,
            needs_drone_assist=needs_drone_assist,
            reacquired_track_id=reacquired_track_id,
        )

    def _transition(self, new_state: TrackingState, now: float) -> None:
        if new_state == self._state:
            return
        logger.info(
            "State change",
            extra={
                "from": self._state.value,
                "to": new_state.value,
                "held_s": round(now - self._state_since, 2),
            },
        )
        self._state = new_state
        self._state_since = now
