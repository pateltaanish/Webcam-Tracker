"""Typed configuration loading (YAML defaults + environment variable overrides)."""

from webcam_tracker.config.settings import (
    PROJECT_ROOT,
    AppConfig,
    DetectionConfig,
    GimbalAxisConfig,
    GimbalConfig,
    MotionPredictionConfig,
    PerfMonitorConfig,
    RecoveryConfig,
    StateMachineConfig,
    TrackingConfig,
    load_config,
)

__all__ = [
    "AppConfig",
    "DetectionConfig",
    "GimbalAxisConfig",
    "GimbalConfig",
    "MotionPredictionConfig",
    "PROJECT_ROOT",
    "PerfMonitorConfig",
    "RecoveryConfig",
    "StateMachineConfig",
    "TrackingConfig",
    "load_config",
]
