"""Builds a MotionPredictor from config."""

from __future__ import annotations

import time
from collections.abc import Callable

from webcam_tracker.config import AppConfig
from webcam_tracker.motion_prediction.predictor import MotionPredictor


def create_motion_predictor(
    config: AppConfig, clock: Callable[[], float] = time.monotonic
) -> MotionPredictor:
    motion = config.motion_prediction
    return MotionPredictor(
        process_noise=motion.process_noise,
        measurement_noise=motion.measurement_noise,
        max_coast_seconds=motion.max_coast_seconds,
        clock=clock,
    )
