"""Unit tests for webcam_tracker.motion_prediction.

The Kalman filter's exact per-step covariance math is impractical to
hand-derive, so these tests assert the *behavior that matters*: velocity
converges toward the truth on constant-velocity input, a stationary point
stays put with ~zero velocity, and pure lookahead doesn't mutate state.
MotionPredictor is driven with a fake clock so dt is exact.
"""

from __future__ import annotations

import pytest

from webcam_tracker.motion_prediction import KalmanFilter2D, MotionPredictor
from webcam_tracker.tracking import TrackedPerson

from ._fake_clock import FakeClock


def _person(track_id: int, cx: float, cy: float, half: float = 10.0) -> TrackedPerson:
    """A tracked box centered on (cx, cy)."""
    return TrackedPerson(
        track_id=track_id,
        x1=cx - half,
        y1=cy - half,
        x2=cx + half,
        y2=cy + half,
        confidence=0.9,
    )


class TestKalmanFilter2D:
    def test_initial_state(self) -> None:
        kalman = KalmanFilter2D((100.0, 50.0), process_noise=50.0, measurement_noise=5.0)
        assert kalman.position == pytest.approx((100.0, 50.0))
        assert kalman.velocity == pytest.approx((0.0, 0.0))

    def test_predict_with_nonpositive_dt_is_noop(self) -> None:
        kalman = KalmanFilter2D((0.0, 0.0), process_noise=50.0, measurement_noise=5.0)
        kalman.predict(0.0)
        kalman.predict(-1.0)
        assert kalman.position == pytest.approx((0.0, 0.0))

    def test_velocity_converges_on_constant_velocity_input(self) -> None:
        # True motion: 100 px/s in x, 0 in y, sampled every 0.1s.
        kalman = KalmanFilter2D((0.0, 0.0), process_noise=50.0, measurement_noise=5.0)
        dt = 0.1
        x = 0.0
        for _ in range(40):
            x += 100.0 * dt
            kalman.predict(dt)
            kalman.correct((x, 0.0))
        vx, vy = kalman.velocity
        assert vx == pytest.approx(100.0, abs=10.0)
        assert vy == pytest.approx(0.0, abs=5.0)

    def test_stationary_point_stays_put(self) -> None:
        kalman = KalmanFilter2D((25.0, 75.0), process_noise=50.0, measurement_noise=5.0)
        for _ in range(20):
            kalman.predict(0.1)
            kalman.correct((25.0, 75.0))
        assert kalman.position == pytest.approx((25.0, 75.0), abs=1.0)
        assert kalman.velocity == pytest.approx((0.0, 0.0), abs=1.0)

    def test_predict_position_is_pure_lookahead(self) -> None:
        kalman = KalmanFilter2D((0.0, 0.0), process_noise=50.0, measurement_noise=5.0)
        dt = 0.1
        x = 0.0
        for _ in range(40):
            x += 100.0 * dt
            kalman.predict(dt)
            kalman.correct((x, 0.0))
        position_before = kalman.position
        velocity_before = kalman.velocity
        ahead = kalman.predict_position(1.0)
        # Extrapolated ~1s at ~100 px/s ahead of current position...
        assert ahead[0] == pytest.approx(position_before[0] + velocity_before[0], abs=1e-6)
        # ...and the call didn't mutate the filter.
        assert kalman.position == pytest.approx(position_before)
        assert kalman.velocity == pytest.approx(velocity_before)


class TestMotionPredictor:
    def test_unknown_track_returns_none(self) -> None:
        predictor = MotionPredictor(50.0, 5.0, 1.5, clock=FakeClock())
        assert predictor.motion(999) is None

    def test_new_track_seeds_at_center_with_zero_velocity(self) -> None:
        clock = FakeClock()
        predictor = MotionPredictor(50.0, 5.0, 1.5, clock=clock)
        predictor.update([_person(1, 100.0, 50.0)])
        motion = predictor.motion(1)
        assert motion is not None
        assert motion.position == pytest.approx((100.0, 50.0))
        assert motion.velocity == pytest.approx((0.0, 0.0))

    def test_velocity_estimated_for_moving_track(self) -> None:
        clock = FakeClock()
        predictor = MotionPredictor(50.0, 5.0, 1.5, clock=clock)
        x = 0.0
        for _ in range(40):
            x += 10.0  # 10 px per frame
            clock.advance(0.1)  # ... every 0.1s -> 100 px/s
            predictor.update([_person(1, x, 50.0)])
        motion = predictor.motion(1)
        assert motion is not None
        assert motion.velocity[0] == pytest.approx(100.0, abs=15.0)
        assert motion.speed == pytest.approx(100.0, abs=15.0)

    def test_lost_track_coasts_then_is_pruned(self) -> None:
        clock = FakeClock()
        predictor = MotionPredictor(50.0, 5.0, max_coast_seconds=1.0, clock=clock)
        predictor.update([_person(1, 10.0, 10.0)])
        clock.advance(0.1)
        predictor.update([_person(1, 20.0, 10.0)])

        # Track disappears. Within the coast window, its motion is still readable.
        clock.advance(0.5)
        predictor.update([])
        assert predictor.motion(1) is not None
        assert 1 in predictor.known_track_ids()

        # Past max_coast_seconds since it was last seen, it's pruned.
        clock.advance(1.0)
        predictor.update([])
        assert predictor.motion(1) is None
        assert 1 not in predictor.known_track_ids()

    def test_reappearing_track_before_prune_keeps_filter(self) -> None:
        clock = FakeClock()
        predictor = MotionPredictor(50.0, 5.0, max_coast_seconds=1.0, clock=clock)
        predictor.update([_person(7, 10.0, 10.0)])
        clock.advance(0.5)
        predictor.update([])  # missed one frame
        clock.advance(0.3)
        predictor.update([_person(7, 12.0, 10.0)])  # back within coast window
        assert predictor.motion(7) is not None
