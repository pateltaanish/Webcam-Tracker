"""Unit tests for webcam_tracker.gimbal_control.

Each PID term (P/I/D) is tested in isolation by zeroing the other gains, so
the expected numbers can be hand-derived exactly rather than approximated.
Uses a fake, manually-advanced clock -- no real sleeping, no flakiness.
"""

from __future__ import annotations

import pytest

from webcam_tracker.gimbal_control import AxisController, GimbalController

from ._fake_clock import FakeClock

# Deliberately huge so these limits never bind unless a test overrides them
# to specifically test that limit.
_UNLIMITED = 1e6


def _make_axis(
    kp: float = 0.0,
    ki: float = 0.0,
    kd: float = 0.0,
    deadband: float = 0.0,
    max_velocity_deg_s: float = _UNLIMITED,
    max_acceleration_deg_s2: float = _UNLIMITED,
    angle_limit_deg: float = _UNLIMITED,
    integral_limit: float = _UNLIMITED,
    clock: FakeClock | None = None,
) -> AxisController:
    return AxisController(
        kp=kp,
        ki=ki,
        kd=kd,
        deadband=deadband,
        max_velocity_deg_s=max_velocity_deg_s,
        max_acceleration_deg_s2=max_acceleration_deg_s2,
        angle_limit_deg=angle_limit_deg,
        integral_limit=integral_limit,
        clock=clock if clock is not None else FakeClock(),
    )


class TestAxisController:
    def test_first_update_returns_zero_velocity(self) -> None:
        clock = FakeClock()
        axis = _make_axis(kp=10.0, clock=clock)
        velocity, angle, saturated = axis.update(0.5)
        assert velocity == 0.0
        assert angle == 0.0
        assert saturated is False

    def test_deadband_zeroes_error_below_threshold(self) -> None:
        clock = FakeClock()
        axis = _make_axis(kp=10.0, deadband=0.05, clock=clock)
        axis.update(0.5)  # warmup
        clock.advance(0.1)
        velocity, _, _ = axis.update(0.03)  # below deadband
        assert velocity == 0.0

    def test_proportional_term(self) -> None:
        clock = FakeClock()
        axis = _make_axis(kp=10.0, clock=clock)
        axis.update(0.5)  # warmup, dt=0
        clock.advance(0.1)
        velocity, angle, _ = axis.update(0.5)
        assert velocity == pytest.approx(5.0)  # kp * error = 10 * 0.5
        assert angle == pytest.approx(0.5)  # velocity * dt = 5.0 * 0.1

    def test_integral_term_accumulates_over_steps(self) -> None:
        clock = FakeClock()
        axis = _make_axis(ki=100.0, clock=clock)
        axis.update(0.5)  # warmup, dt=0 -> integral stays 0
        clock.advance(0.1)
        v1, _, _ = axis.update(0.5)  # integral = 0.5*0.1 = 0.05 -> v = 100*0.05
        assert v1 == pytest.approx(5.0)
        clock.advance(0.1)
        v2, _, _ = axis.update(0.5)  # integral = 0.10 -> v = 100*0.10
        assert v2 == pytest.approx(10.0)

    def test_integral_anti_windup_clamp(self) -> None:
        clock = FakeClock()
        axis = _make_axis(ki=100.0, integral_limit=0.08, clock=clock)
        axis.update(0.5)  # warmup
        clock.advance(0.1)
        axis.update(0.5)  # integral = 0.05, under the 0.08 limit
        clock.advance(0.1)
        v, _, _ = axis.update(0.5)  # unclamped integral would be 0.10
        assert v == pytest.approx(8.0)  # clamped: 100 * 0.08

    def test_derivative_term_reacts_to_error_change(self) -> None:
        clock = FakeClock()
        axis = _make_axis(kd=2.0, clock=clock)
        axis.update(0.0)  # warmup, previous_error=0.0
        clock.advance(0.1)
        v1, _, _ = axis.update(0.5)  # derivative = (0.5-0.0)/0.1 = 5.0
        assert v1 == pytest.approx(10.0)  # kd * derivative = 2 * 5.0
        clock.advance(0.1)
        v2, _, _ = axis.update(0.5)  # error unchanged -> derivative = 0
        assert v2 == pytest.approx(0.0)

    def test_velocity_clamped_to_max_and_flags_saturated(self) -> None:
        clock = FakeClock()
        axis = _make_axis(kp=1000.0, max_velocity_deg_s=50.0, clock=clock)
        axis.update(0.5)  # warmup
        clock.advance(0.1)
        velocity, angle, saturated = axis.update(0.5)
        assert velocity == pytest.approx(50.0)
        assert saturated is True
        assert angle == pytest.approx(5.0)  # 50 * 0.1

    def test_acceleration_rate_limits_without_flagging_saturated(self) -> None:
        # Rate-limiting is a transient "still ramping up" state, not a hard
        # ceiling -- it must NOT set `saturated` (that's reserved for
        # velocity/angle limits the axis can never exceed). See
        # AxisController.update()'s docstring.
        clock = FakeClock()
        axis = _make_axis(
            kp=1000.0, max_velocity_deg_s=1000.0, max_acceleration_deg_s2=10.0, clock=clock
        )
        axis.update(0.5)  # warmup
        clock.advance(0.1)
        velocity, _, saturated = axis.update(0.5)
        # Raw P output would be 500; rate limit caps the change to
        # accel*dt = 10*0.1 = 1.0 from the previous velocity of 0.
        assert velocity == pytest.approx(1.0)
        assert saturated is False

    def test_angle_clamped_and_flagged_saturated_independent_of_velocity(self) -> None:
        clock = FakeClock()
        axis = _make_axis(kp=5.0, max_velocity_deg_s=100.0, angle_limit_deg=0.1, clock=clock)
        axis.update(0.5)  # warmup
        clock.advance(0.1)
        velocity, angle, saturated = axis.update(0.5)
        assert velocity == pytest.approx(2.5)  # well under max_velocity=100
        assert angle == pytest.approx(0.1)  # clamped from 2.5*0.1=0.25
        assert saturated is True

    def test_hold_zeroes_velocity_and_freezes_angle(self) -> None:
        clock = FakeClock()
        axis = _make_axis(kp=10.0, clock=clock)
        axis.update(0.5)  # warmup
        clock.advance(0.1)
        axis.update(0.5)  # velocity=5.0, angle=0.5
        angle_before = axis.angle_deg

        velocity, angle, saturated = axis.hold()

        assert velocity == 0.0
        assert angle == angle_before
        assert saturated is False

    def test_reset_clears_integral_and_velocity_but_not_angle(self) -> None:
        clock = FakeClock()
        axis = _make_axis(ki=100.0, clock=clock)
        axis.update(0.5)  # warmup
        clock.advance(0.1)
        axis.update(0.5)  # integral=0.05
        angle_before = axis.angle_deg

        axis.reset()

        assert axis.angle_deg == angle_before
        clock.advance(0.1)
        v, _, _ = axis.update(0.5)  # integral starts fresh: 0.5*0.1=0.05 -> v=5.0, not 10.0
        assert v == pytest.approx(5.0)


def _make_gimbal(clock: FakeClock) -> GimbalController:
    pan = _make_axis(kp=1.0, clock=clock)
    tilt = _make_axis(kp=1.0, clock=clock)
    return GimbalController(pan=pan, tilt=tilt)


class TestGimbalController:
    def test_none_error_marks_target_not_visible_and_holds(self) -> None:
        clock = FakeClock()
        gimbal = _make_gimbal(clock)
        command = gimbal.compute(None, None)
        assert command.target_visible is False
        assert command.pan_velocity_deg_s == 0.0
        assert command.tilt_velocity_deg_s == 0.0

    def test_valid_error_marks_target_visible(self) -> None:
        clock = FakeClock()
        gimbal = _make_gimbal(clock)
        command = gimbal.compute(0.5, -0.3)
        assert command.target_visible is True

    def test_produces_expected_velocity_per_axis_after_warmup(self) -> None:
        clock = FakeClock()
        gimbal = _make_gimbal(clock)
        gimbal.compute(0.5, -0.3)  # warmup
        clock.advance(0.1)
        command = gimbal.compute(0.5, -0.3)
        assert command.pan_velocity_deg_s == pytest.approx(0.5)  # kp=1.0
        assert command.tilt_velocity_deg_s == pytest.approx(-0.3)

    def test_emergency_stop_forces_hold_even_with_valid_error(self) -> None:
        clock = FakeClock()
        gimbal = _make_gimbal(clock)
        gimbal.compute(0.5, -0.3)
        clock.advance(0.1)
        gimbal.compute(0.5, -0.3)  # establishes nonzero velocity

        gimbal.emergency_stop()
        clock.advance(0.1)
        command = gimbal.compute(0.5, -0.3)

        assert command.pan_velocity_deg_s == 0.0
        assert command.tilt_velocity_deg_s == 0.0
        assert command.emergency_stopped is True
        assert gimbal.is_emergency_stopped is True

    def test_resume_restores_normal_operation(self) -> None:
        clock = FakeClock()
        gimbal = _make_gimbal(clock)
        gimbal.emergency_stop()
        gimbal.resume()
        assert gimbal.is_emergency_stopped is False

        gimbal.compute(0.5, -0.3)  # warmup
        clock.advance(0.1)
        command = gimbal.compute(0.5, -0.3)
        assert command.pan_velocity_deg_s == pytest.approx(0.5)

    def test_saturated_flags_propagate_independently_per_axis(self) -> None:
        clock = FakeClock()
        pan = _make_axis(kp=1000.0, max_velocity_deg_s=10.0, clock=clock)
        tilt = _make_axis(kp=1.0, clock=clock)
        gimbal = GimbalController(pan=pan, tilt=tilt)

        gimbal.compute(0.5, 0.5)  # warmup
        clock.advance(0.1)
        command = gimbal.compute(0.5, 0.5)

        assert command.pan_saturated is True
        assert command.tilt_saturated is False
