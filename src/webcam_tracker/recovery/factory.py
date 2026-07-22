"""Builds a RecoveryController from config."""

from __future__ import annotations

import time
from collections.abc import Callable

from webcam_tracker.config import AppConfig
from webcam_tracker.recovery.controller import RecoveryController


def create_recovery_controller(
    config: AppConfig, clock: Callable[[], float] = time.monotonic
) -> RecoveryController:
    recovery = config.recovery
    return RecoveryController(
        give_up_seconds=recovery.give_up_seconds,
        prediction_horizon_seconds=recovery.prediction_horizon_seconds,
        reacquire_radius_fraction=recovery.reacquire_radius_fraction,
        search_amplitude=recovery.search_amplitude,
        search_period_seconds=recovery.search_period_seconds,
        clock=clock,
    )
