"""Builds a fully-wired TrackingStateMachine from config.

Assembles the four decision components (selector, motion predictor, recovery
controller, gimbal controller) and hands them to the state machine. Perception
(detector, tracker, video source) is built separately by the caller and feeds
the state machine's `update()` -- the state machine coordinates decisions, not
perception.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from webcam_tracker.config import AppConfig
from webcam_tracker.gimbal_control import create_gimbal_controller
from webcam_tracker.motion_prediction import create_motion_predictor
from webcam_tracker.recovery import create_recovery_controller
from webcam_tracker.state_machine.state_machine import TrackingStateMachine
from webcam_tracker.target_selection import TargetSelector


def create_state_machine(
    config: AppConfig, clock: Callable[[], float] = time.monotonic
) -> TrackingStateMachine:
    """Build a fully-wired state machine. `clock` is threaded to every
    time-based component (predictor, recovery, gimbal, and the state machine
    itself) so callers can drive the whole pipeline off a deterministic or a
    video-time clock instead of the wall clock -- used by the integration
    tests (fake clock) and the integration report tool (video-time clock)."""
    return TrackingStateMachine(
        selector=TargetSelector(),
        predictor=create_motion_predictor(config, clock=clock),
        recovery=create_recovery_controller(config, clock=clock),
        gimbal=create_gimbal_controller(config, clock=clock),
        occlusion_timeout_seconds=config.state_machine.occlusion_timeout_seconds,
        clock=clock,
    )
