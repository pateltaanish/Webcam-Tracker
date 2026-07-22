"""Single-axis PID controller with deadband, rate limiting, angle clamping,
and anti-windup. Used twice by GimbalController (once for pan, once for
tilt) -- everything axis-specific lives here so GimbalController only has to
combine two of these plus target-visibility/e-stop handling.

Background for anyone new to control theory: a PID controller turns an
*error* (how far off from where you want to be) into a *correction* (here,
an angular velocity command) using three terms:

  - P (proportional): correction proportional to the CURRENT error. Bigger
    error -> bigger correction. Alone, this tends to leave a small steady
    "off by a bit" error, because the correction shrinks to nothing as the
    error shrinks to nothing.
  - I (integral): correction proportional to the ACCUMULATED error over
    time. This is what eliminates that steady-state offset -- a small
    persistent error keeps building up the integral term until it's enough
    to close the gap. Left unchecked it can also cause overshoot ("windup"),
    which is why there's an integral_limit clamp below.
  - D (derivative): correction proportional to how fast the error is
    CHANGING. This dampens the response -- it pushes back against fast
    swings, reducing overshoot/oscillation from the P and I terms.

output = kp*error + ki*integral(error) + kd*d(error)/dt
"""

from __future__ import annotations

import time
from collections.abc import Callable

from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class AxisController:
    """PID control for one gimbal axis, plus a simulated absolute angle.

    `update(error)` is the normal path: runs the PID math and integrates the
    resulting velocity into a simulated position. `hold()` is the "target
    not visible / emergency stopped" path: zero velocity, position frozen
    exactly where it was.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        deadband: float,
        max_velocity_deg_s: float,
        max_acceleration_deg_s2: float,
        angle_limit_deg: float,
        integral_limit: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._deadband = deadband
        self._max_velocity = max_velocity_deg_s
        self._max_acceleration = max_acceleration_deg_s2
        self._angle_limit = angle_limit_deg
        self._integral_limit = integral_limit
        self._clock = clock

        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_velocity = 0.0
        self._angle = 0.0
        self._last_update_time: float | None = None

    @property
    def angle_deg(self) -> float:
        return self._angle

    def update(self, error: float) -> tuple[float, float, bool]:
        """Advance the controller by one time step given the current error.

        Returns (velocity_deg_s, angle_deg, saturated) -- `saturated` is
        True if the commanded velocity or the resulting angle hit its clamp
        (the signal drone_control will eventually use to know the gimbal
        alone can't correct further).

        Note: the very first update() call after construction (or after a
        gap with no calls) always returns velocity=0 -- with no previous
        call, the elapsed time (dt) is unknown, so rate limiting has nothing
        to measure "per second" against and conservatively allows no
        change. In a real continuous loop this is a harmless one-frame
        startup artifact; don't mistake it for the controller being broken
        if you call update() once in isolation.
        """
        dt = self._advance_clock()

        effective_error = 0.0 if abs(error) < self._deadband else error

        # Integral term with anti-windup: clamp the accumulator itself so a
        # long period of persistent error can't build up a value so large
        # that, once the error clears, the controller massively overshoots
        # correcting for it.
        self._integral = _clamp(
            self._integral + effective_error * dt, -self._integral_limit, self._integral_limit
        )

        derivative = (effective_error - self._previous_error) / dt if dt > 0 else 0.0
        self._previous_error = effective_error

        raw_velocity = (
            self._kp * effective_error + self._ki * self._integral + self._kd * derivative
        )

        # Rate limiting: cap how much the commanded velocity can change in
        # this time step, so commands ramp smoothly instead of jumping.
        max_delta = self._max_acceleration * dt
        velocity = _clamp(
            raw_velocity, self._previous_velocity - max_delta, self._previous_velocity + max_delta
        )

        velocity_saturated = abs(velocity) > self._max_velocity
        velocity = _clamp(velocity, -self._max_velocity, self._max_velocity)
        self._previous_velocity = velocity

        candidate_angle = self._angle + velocity * dt
        angle_saturated = abs(candidate_angle) > self._angle_limit
        self._angle = _clamp(candidate_angle, -self._angle_limit, self._angle_limit)

        return velocity, self._angle, (velocity_saturated or angle_saturated)

    def hold(self) -> tuple[float, float, bool]:
        """Zero velocity, angle unchanged. Still advances the internal clock
        so the next real update()'s dt is measured correctly."""
        self._advance_clock()
        self._previous_velocity = 0.0
        return 0.0, self._angle, False

    def reset(self) -> None:
        """Zero the integral/velocity state (e.g. on emergency stop) without
        moving the simulated angle."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_velocity = 0.0

    def _advance_clock(self) -> float:
        now = self._clock()
        dt = 0.0 if self._last_update_time is None else max(now - self._last_update_time, 0.0)
        self._last_update_time = now
        return dt
