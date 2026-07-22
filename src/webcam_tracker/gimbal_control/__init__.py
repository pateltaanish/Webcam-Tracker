"""Gimbal command generation (simulated -- Stage 1.7).

Converts target pixel/normalized error (from target_selection) into
pan/tilt velocity commands via PID control, with deadband, rate limiting,
angle clamping, and anti-windup. This never drives real hardware: there is
no physical gimbal in this project yet, and building the control logic now
-- against real tracking data -- lets it be designed and tested before any
hardware exists. A real motor driver slots in behind the same
GimbalController/GimbalCommand interface later (Stage 3) without changing
this module or anything upstream of it.

Scope: this module only computes commands from the error it's given. It has
no opinion about search patterns or what to do when there's no target for a
long time -- that policy belongs to the state_machine (Stage 1.9).
"""

from webcam_tracker.gimbal_control.axis_controller import AxisController
from webcam_tracker.gimbal_control.factory import create_gimbal_controller
from webcam_tracker.gimbal_control.gimbal_controller import GimbalCommand, GimbalController

__all__ = ["AxisController", "GimbalCommand", "GimbalController", "create_gimbal_controller"]
