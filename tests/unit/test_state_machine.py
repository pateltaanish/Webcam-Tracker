"""Unit tests for webcam_tracker.state_machine.

Wires the state machine up with the REAL decision components (selector,
predictor, recovery, gimbal) -- they're all pure logic -- sharing one fake
clock, so a whole scenario (track -> occlude -> search -> give up, or
-> reacquire) can be driven frame by frame with exact timing.
"""

from __future__ import annotations

import numpy as np
import pytest

from webcam_tracker.gimbal_control import AxisController, GimbalController
from webcam_tracker.motion_prediction import MotionPredictor
from webcam_tracker.recovery import RecoveryController
from webcam_tracker.state_machine import TrackingState, TrackingStateMachine
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson

from ._fake_clock import FakeClock

_IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)

_W = 100
_H = 100
_BIG = 1e6


def _person(track_id: int, cx: float, cy: float, half: float = 10.0) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id, x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half, confidence=0.9
    )


def _axis(clock: FakeClock, max_velocity: float) -> AxisController:
    return AxisController(
        kp=80.0,
        ki=0.0,
        kd=0.0,
        deadband=0.0,
        max_velocity_deg_s=max_velocity,
        max_acceleration_deg_s2=_BIG,
        angle_limit_deg=_BIG,
        integral_limit=_BIG,
        clock=clock,
    )


def _make(
    clock: FakeClock,
    occlusion_timeout: float = 1.0,
    give_up: float = 5.0,
    gimbal_max_velocity: float = _BIG,
    identity: object | None = None,
) -> TrackingStateMachine:
    return TrackingStateMachine(
        selector=TargetSelector(),
        predictor=MotionPredictor(50.0, 5.0, 1.5, clock=clock),
        recovery=RecoveryController(give_up, 1.0, 0.3, 0.5, 2.0, clock=clock),
        gimbal=GimbalController(
            pan=_axis(clock, gimbal_max_velocity), tilt=_axis(clock, gimbal_max_velocity)
        ),
        occlusion_timeout_seconds=occlusion_timeout,
        identity=identity,  # type: ignore[arg-type]  # tests pass a duck-typed fake
        clock=clock,
    )


class _FakeIdentity:
    """Duck-typed stand-in for IdentityTracker. `resolution` maps person_id ->
    the track_id that should currently be resolved (or None)."""

    def __init__(self) -> None:
        self.resolution: dict[str, int | None] = {}
        self.update_calls = 0

    def update(self, image: object, tracked_people: object) -> None:
        self.update_calls += 1

    def resolve_person(self, person_id: str) -> int | None:
        return self.resolution.get(person_id)


class TestTrackingStateMachine:
    def test_idle_with_no_selection(self) -> None:
        clock = FakeClock()
        sm = _make(clock)
        clock.advance(0.1)
        status = sm.update([_person(1, 50, 50)], _W, _H)
        assert status.state is TrackingState.IDLE

    def test_tracking_when_target_visible(self) -> None:
        clock = FakeClock()
        sm = _make(clock)
        sm.selector.select(1)
        clock.advance(0.1)
        status = sm.update([_person(1, 50, 50)], _W, _H)
        assert status.state is TrackingState.TRACKING
        assert status.target_status is not None
        assert status.target_status.visible

    def test_occluded_then_recovery_search(self) -> None:
        clock = FakeClock()
        sm = _make(clock, occlusion_timeout=1.0)
        sm.selector.select(1)
        clock.advance(0.1)
        sm.update([_person(1, 50, 50)], _W, _H)  # TRACKING

        clock.advance(0.1)
        occluded = sm.update([], _W, _H)  # first missing frame
        assert occluded.state is TrackingState.TEMPORARILY_OCCLUDED

        clock.advance(1.2)  # past occlusion timeout
        searching = sm.update([], _W, _H)
        assert searching.state is TrackingState.RECOVERY_SEARCH
        assert searching.recovery_status.search_error is not None

    def test_same_id_return_during_occlusion_resumes_tracking(self) -> None:
        clock = FakeClock()
        sm = _make(clock, occlusion_timeout=1.0)
        sm.selector.select(1)
        clock.advance(0.1)
        sm.update([_person(1, 50, 50)], _W, _H)
        clock.advance(0.1)
        sm.update([], _W, _H)  # occluded
        clock.advance(0.3)  # still within occlusion window
        back = sm.update([_person(1, 55, 50)], _W, _H)
        assert back.state is TrackingState.TRACKING

    def test_occlusion_does_not_grab_a_different_person(self) -> None:
        clock = FakeClock()
        sm = _make(clock, occlusion_timeout=1.0)
        sm.selector.select(1)
        clock.advance(0.1)
        sm.update([_person(1, 50, 50)], _W, _H)
        clock.advance(0.1)
        # Target gone, but a different new person appears right where it was.
        status = sm.update([_person(2, 50, 50)], _W, _H)
        assert status.state is TrackingState.TEMPORARILY_OCCLUDED
        assert sm.selector.target_id == 1  # did NOT re-lock onto id 2
        assert status.reacquired_track_id is None

    def test_recovery_search_reacquires_new_track(self) -> None:
        clock = FakeClock()
        sm = _make(clock, occlusion_timeout=1.0)
        sm.selector.select(1)
        clock.advance(0.1)
        sm.update([_person(1, 50, 50)], _W, _H)
        clock.advance(0.1)
        sm.update([], _W, _H)  # occluded
        clock.advance(1.2)
        sm.update([], _W, _H)  # RECOVERY_SEARCH
        clock.advance(0.1)
        relock = sm.update([_person(2, 52, 50)], _W, _H)  # new track near predicted
        assert relock.state is TrackingState.TRACKING
        assert relock.reacquired_track_id == 2
        assert sm.selector.target_id == 2

    def test_gives_up_to_safe_hover(self) -> None:
        clock = FakeClock()
        sm = _make(clock, occlusion_timeout=1.0, give_up=2.0)
        sm.selector.select(1)
        clock.advance(0.1)
        sm.update([_person(1, 50, 50)], _W, _H)
        clock.advance(0.1)
        sm.update([], _W, _H)  # occluded
        clock.advance(1.2)
        sm.update([], _W, _H)  # RECOVERY_SEARCH begins
        clock.advance(2.5)  # past give_up (measured from search start)
        hover = sm.update([], _W, _H)
        assert hover.state is TrackingState.SAFE_HOVER_REQUESTED

    def test_emergency_stop_and_resume(self) -> None:
        clock = FakeClock()
        sm = _make(clock)
        sm.selector.select(1)
        clock.advance(0.1)
        assert sm.update([_person(1, 50, 50)], _W, _H).state is TrackingState.TRACKING

        sm.emergency_stop()
        clock.advance(0.1)
        assert sm.update([_person(1, 50, 50)], _W, _H).state is TrackingState.STOPPED

        sm.resume()
        clock.advance(0.1)
        assert sm.update([_person(1, 50, 50)], _W, _H).state is TrackingState.TRACKING

    def test_needs_drone_assist_when_gimbal_saturates(self) -> None:
        clock = FakeClock()
        sm = _make(clock, gimbal_max_velocity=1.0)  # tiny -> saturates immediately
        sm.selector.select(1)
        # Target far off-center (pan error 0.8) -> commanded velocity >> 1.0.
        clock.advance(0.1)
        first = sm.update([_person(1, 90, 50)], _W, _H)
        assert first.needs_drone_assist is False  # first gimbal frame: velocity 0
        clock.advance(0.1)
        second = sm.update([_person(1, 90, 50)], _W, _H)
        assert second.state is TrackingState.TRACKING
        assert second.needs_drone_assist is True
        assert second.gimbal_command.pan_saturated is True

    def test_time_in_state_resets_on_transition(self) -> None:
        clock = FakeClock()
        sm = _make(clock)
        sm.selector.select(1)
        clock.advance(0.1)
        first = sm.update([_person(1, 50, 50)], _W, _H)  # IDLE -> TRACKING transition
        assert first.time_in_state_s == pytest.approx(0.0)
        clock.advance(0.1)
        second = sm.update([_person(1, 50, 50)], _W, _H)  # still TRACKING
        assert second.time_in_state_s == pytest.approx(0.1)


class TestIdentityMode:
    def test_select_person_without_identity_raises(self) -> None:
        sm = _make(FakeClock())  # no identity tracker
        with pytest.raises(RuntimeError):
            sm.select_person("alice")

    def test_locks_onto_identity_resolved_track(self) -> None:
        clock = FakeClock()
        identity = _FakeIdentity()
        identity.resolution["alice"] = 5
        sm = _make(clock, identity=identity)
        sm.select_person("alice")

        clock.advance(0.1)
        status = sm.update([_person(5, 50, 50)], _W, _H, image=_IMAGE)
        assert status.state is TrackingState.TRACKING
        assert sm.selector.target_id == 5
        assert identity.update_calls == 1

    def test_reacquires_person_across_track_id_change(self) -> None:
        clock = FakeClock()
        identity = _FakeIdentity()
        identity.resolution["alice"] = 1
        sm = _make(clock, identity=identity)
        sm.select_person("alice")

        clock.advance(0.1)
        assert sm.update([_person(1, 50, 50)], _W, _H, image=_IMAGE).state is TrackingState.TRACKING
        assert sm.selector.target_id == 1

        # Person leaves and returns as a brand-new track id; identity resolves
        # them to it -> re-lock by face, no geometry involved.
        identity.resolution["alice"] = 9
        clock.advance(0.1)
        status = sm.update([_person(9, 60, 50)], _W, _H, image=_IMAGE)
        assert status.state is TrackingState.TRACKING
        assert sm.selector.target_id == 9

    def test_geometric_reacquire_suppressed_in_identity_mode(self) -> None:
        clock = FakeClock()
        identity = _FakeIdentity()
        identity.resolution["alice"] = 1
        sm = _make(clock, occlusion_timeout=1.0, identity=identity)
        sm.select_person("alice")

        clock.advance(0.1)
        sm.update([_person(1, 50, 50)], _W, _H, image=_IMAGE)  # TRACKING id 1

        # Alice can no longer be identified; her track is gone.
        identity.resolution["alice"] = None
        clock.advance(0.1)
        sm.update([], _W, _H, image=_IMAGE)  # TEMPORARILY_OCCLUDED
        clock.advance(1.2)
        sm.update([], _W, _H, image=_IMAGE)  # RECOVERY_SEARCH

        # A DIFFERENT person appears right where Alice was predicted. In Stage 1
        # (geometric) this would be grabbed; in identity mode it must NOT be.
        clock.advance(0.1)
        status = sm.update([_person(2, 52, 50)], _W, _H, image=_IMAGE)
        assert status.reacquired_track_id is None
        assert sm.selector.target_id == 1  # still the (absent) original, never 2
        assert status.state is not TrackingState.TRACKING
