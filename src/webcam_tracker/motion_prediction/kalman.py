"""A constant-velocity Kalman filter for one 2D point, from scratch.

Built from first principles (like gimbal_control's PID) rather than pulled
from a library, so the behavior is fully understood and testable.

Background for anyone new to Kalman filters: a Kalman filter estimates the
true state of something (here, a tracked person's box-center position AND
velocity) from noisy measurements (here, the detector's per-frame box, which
jitters a few pixels even on a still person). Each step is two phases:

  - predict: roll the state forward in time using a motion model. Our model
    is "constant velocity": assume the person keeps moving at their current
    estimated speed. This also grows our uncertainty (we're guessing).
  - correct (a.k.a. update): fold in the new measurement, pulling the
    estimate toward it by an amount (the "Kalman gain") that depends on how
    much we trust the measurement vs. the model. This shrinks uncertainty.

Why we bother instead of just differencing positions: the detector only ever
tells us *where* a box is, never how fast it's moving. The filter infers
velocity from how position changes over time, and smooths out the jitter --
and that inferred velocity is exactly what the recovery module needs to guess
which way a just-lost target was heading.

State vector: [x, y, vx, vy] -- position (px) and velocity (px/s).
We measure only position, so the measurement is [x, y].
"""

from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """Constant-velocity Kalman filter tracking one 2D point in pixel space."""

    def __init__(
        self,
        initial_position: tuple[float, float],
        process_noise: float,
        measurement_noise: float,
    ) -> None:
        px, py = initial_position
        # Start at the first measured position, velocity unknown (assume zero).
        self._x = np.array([px, py, 0.0, 0.0], dtype=float)
        # Initial covariance (uncertainty): we just measured position so it's
        # fairly certain, but we have no idea about velocity yet -> large.
        self._P = np.diag([measurement_noise, measurement_noise, 1000.0, 1000.0]).astype(float)
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        # Measurement matrix: we observe position (rows 0,1), not velocity.
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    @property
    def position(self) -> tuple[float, float]:
        return float(self._x[0]), float(self._x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self._x[2]), float(self._x[3])

    def predict(self, dt: float) -> None:
        """Roll the estimate forward by dt seconds under constant velocity,
        growing uncertainty by the process noise. No-op for dt <= 0."""
        if dt <= 0.0:
            return
        # State transition: x += vx*dt, y += vy*dt, velocities unchanged.
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Discrete white-noise-acceleration process noise: models the fact
        # that real people accelerate (break the constant-velocity assumption)
        # by a random amount each step. process_noise is that acceleration
        # variance; the dt powers spread it across position/velocity correctly.
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self._process_noise
        process_cov = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ]
        )
        self._x = transition @ self._x
        self._P = transition @ self._P @ transition.T + process_cov

    def correct(self, position: tuple[float, float]) -> None:
        """Fold a new measured position into the estimate."""
        measurement = np.array([position[0], position[1]], dtype=float)
        measurement_cov = self._measurement_noise * np.eye(2)
        # Innovation: how far the measurement is from our prediction.
        innovation = measurement - self._H @ self._x
        innovation_cov = self._H @ self._P @ self._H.T + measurement_cov
        # Kalman gain: how much to trust this measurement vs. the model.
        gain = self._P @ self._H.T @ np.linalg.inv(innovation_cov)
        self._x = self._x + gain @ innovation
        self._P = (np.eye(4) - gain @ self._H) @ self._P

    def predict_position(self, seconds_ahead: float) -> tuple[float, float]:
        """Extrapolate where the point will be `seconds_ahead` from now under
        constant velocity, WITHOUT mutating filter state -- a pure lookahead
        for recovery to guess where a lost target went."""
        x, y = self.position
        vx, vy = self.velocity
        return x + vx * seconds_ahead, y + vy * seconds_ahead
