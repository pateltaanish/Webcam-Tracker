"""Builds a GimbalController from config."""

from __future__ import annotations

import time
from collections.abc import Callable

from webcam_tracker.config import AppConfig
from webcam_tracker.gimbal_control.axis_controller import AxisController
from webcam_tracker.gimbal_control.gimbal_controller import GimbalController


def create_gimbal_controller(
    config: AppConfig, clock: Callable[[], float] = time.monotonic
) -> GimbalController:
    gimbal = config.gimbal
    pan = AxisController(
        kp=gimbal.pan.kp,
        ki=gimbal.pan.ki,
        kd=gimbal.pan.kd,
        deadband=gimbal.deadband,
        max_velocity_deg_s=gimbal.max_velocity_deg_s,
        max_acceleration_deg_s2=gimbal.max_acceleration_deg_s2,
        angle_limit_deg=gimbal.pan.angle_limit_deg,
        integral_limit=gimbal.integral_limit,
        clock=clock,
    )
    tilt = AxisController(
        kp=gimbal.tilt.kp,
        ki=gimbal.tilt.ki,
        kd=gimbal.tilt.kd,
        deadband=gimbal.deadband,
        max_velocity_deg_s=gimbal.max_velocity_deg_s,
        max_acceleration_deg_s2=gimbal.max_acceleration_deg_s2,
        angle_limit_deg=gimbal.tilt.angle_limit_deg,
        integral_limit=gimbal.integral_limit,
        clock=clock,
    )
    return GimbalController(pan=pan, tilt=tilt)
