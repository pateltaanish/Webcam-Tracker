"""Unit tests for webcam_tracker.recovery.

Driven with a fake clock so the search/give-up timing is exact. TargetStatus
inputs are produced through a real TargetSelector (its visible/not-visible
logic is what recovery keys off), and target motion is supplied as an
explicit TrackMotion so the predicted-position math is exactly checkable.
"""

from __future__ import annotations

import pytest

from webcam_tracker.motion_prediction import TrackMotion
from webcam_tracker.recovery import RecoveryController, RecoveryState
from webcam_tracker.target_selection import TargetSelector
from webcam_tracker.tracking import TrackedPerson

from ._fake_clock import FakeClock

_W = 100
_H = 100


def _person(track_id: int, cx: float, cy: float, half: float = 10.0) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        x1=cx - half,
        y1=cy - half,
        x2=cx + half,
        y2=cy + half,
        confidence=0.9,
    )


def _make_controller(clock: FakeClock, give_up_seconds: float = 1.0) -> RecoveryController:
    return RecoveryController(
        give_up_seconds=give_up_seconds,
        prediction_horizon_seconds=1.0,
        reacquire_radius_fraction=0.3,  # 0.3 * hypot(100,100) ~= 42px radius
        search_amplitude=0.5,
        search_period_seconds=2.0,
        clock=clock,
    )


class TestRecoveryController:
    def test_no_target_is_idle(self) -> None:
        controller = _make_controller(FakeClock())
        status = controller.update(None, [], None, _W, _H)
        assert status.state is RecoveryState.IDLE

    def test_visible_target_is_tracking(self) -> None:
        controller = _make_controller(FakeClock())
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)
        target_status = selector.status([person], _W, _H)
        status = controller.update(target_status, [person], None, _W, _H)
        assert status.state is RecoveryState.TRACKING

    def test_lost_target_starts_searching(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock)
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)

        controller.update(selector.status([person], _W, _H), [person], None, _W, _H)
        # Person vanishes.
        lost = controller.update(selector.status([], _W, _H), [], None, _W, _H)
        assert lost.state is RecoveryState.SEARCHING
        assert lost.search_error is not None
        assert lost.predicted_position is not None
        pan, tilt = lost.search_error
        assert -1.0 <= pan <= 1.0
        assert -1.0 <= tilt <= 1.0

    def test_predicted_position_extrapolates_last_velocity(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock)
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)
        motion = TrackMotion(track_id=1, position=(50.0, 50.0), velocity=(20.0, 0.0))

        controller.update(selector.status([person], _W, _H), [person], motion, _W, _H)
        controller.update(selector.status([], _W, _H), [], None, _W, _H)  # loss, elapsed=0
        clock.advance(0.5)
        status = controller.update(selector.status([], _W, _H), [], None, _W, _H)
        assert status.predicted_position is not None
        # 50 + 20 px/s * 0.5s = 60
        assert status.predicted_position[0] == pytest.approx(60.0, abs=1e-6)
        assert status.predicted_position[1] == pytest.approx(50.0, abs=1e-6)

    def test_prediction_capped_at_horizon(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock)
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)
        motion = TrackMotion(track_id=1, position=(50.0, 50.0), velocity=(20.0, 0.0))

        controller.update(selector.status([person], _W, _H), [person], motion, _W, _H)
        controller.update(selector.status([], _W, _H), [], None, _W, _H)
        clock.advance(0.9)  # past give_up? no, give_up=1.0; horizon=1.0
        status = controller.update(selector.status([], _W, _H), [], None, _W, _H)
        # elapsed 0.9 < horizon 1.0 -> 50 + 20*0.9 = 68
        assert status.predicted_position is not None
        assert status.predicted_position[0] == pytest.approx(68.0, abs=1e-6)

    def test_reacquires_nearby_new_track(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock)
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)

        controller.update(selector.status([person], _W, _H), [person], None, _W, _H)
        controller.update(selector.status([], _W, _H), [], None, _W, _H)  # lost -> searching
        # A brand-new track (id 2) appears right where the target was.
        newcomer = _person(2, 52.0, 50.0)
        status = controller.update(selector.status([], _W, _H), [newcomer], None, _W, _H)
        assert status.state is RecoveryState.REACQUIRED
        assert status.reacquire_track_id == 2

    def test_does_not_reacquire_track_present_at_loss(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock)
        selector = TargetSelector()
        selector.select(1)
        target = _person(1, 50.0, 50.0)
        bystander = _person(9, 55.0, 50.0)  # on screen at loss, near predicted

        controller.update(
            selector.status([target, bystander], _W, _H), [target, bystander], None, _W, _H
        )
        # Target vanishes; bystander (id 9, known at loss) remains near center.
        status = controller.update(selector.status([bystander], _W, _H), [bystander], None, _W, _H)
        assert status.state is RecoveryState.SEARCHING
        assert status.reacquire_track_id is None

    def test_does_not_reacquire_far_new_track(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock)
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)

        controller.update(selector.status([person], _W, _H), [person], None, _W, _H)
        controller.update(selector.status([], _W, _H), [], None, _W, _H)
        # New track appears far from predicted center (distance 49 > ~42 radius).
        faraway = _person(2, 99.0, 50.0)
        status = controller.update(selector.status([], _W, _H), [faraway], None, _W, _H)
        assert status.state is RecoveryState.SEARCHING
        assert status.reacquire_track_id is None

    def test_gives_up_after_timeout_and_stays(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock, give_up_seconds=1.0)
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)

        controller.update(selector.status([person], _W, _H), [person], None, _W, _H)
        controller.update(selector.status([], _W, _H), [], None, _W, _H)  # searching
        clock.advance(1.5)  # past give_up_seconds
        gave_up = controller.update(selector.status([], _W, _H), [], None, _W, _H)
        assert gave_up.state is RecoveryState.GAVE_UP
        # A new track appearing now is ignored -- we already gave up.
        clock.advance(0.1)
        newcomer = _person(2, 50.0, 50.0)
        still = controller.update(selector.status([], _W, _H), [newcomer], None, _W, _H)
        assert still.state is RecoveryState.GAVE_UP
        assert still.reacquire_track_id is None

    def test_reacquire_then_tracking_on_reselect(self) -> None:
        clock = FakeClock()
        controller = _make_controller(clock)
        selector = TargetSelector()
        selector.select(1)
        person = _person(1, 50.0, 50.0)

        controller.update(selector.status([person], _W, _H), [person], None, _W, _H)
        controller.update(selector.status([], _W, _H), [], None, _W, _H)
        newcomer = _person(2, 50.0, 50.0)
        status = controller.update(selector.status([], _W, _H), [newcomer], None, _W, _H)
        assert status.reacquire_track_id == 2

        # Caller acts on the recommendation.
        selector.select(2)
        back = controller.update(selector.status([newcomer], _W, _H), [newcomer], None, _W, _H)
        assert back.state is RecoveryState.TRACKING
