"""Typed configuration loading (YAML defaults + environment variable overrides)."""

from webcam_tracker.config.settings import (
    PROJECT_ROOT,
    AppConfig,
    DetectionConfig,
    FaceConfig,
    GimbalAxisConfig,
    GimbalConfig,
    IdentityConfig,
    MotionPredictionConfig,
    PerfMonitorConfig,
    RecoveryConfig,
    RegistrationConfig,
    StateMachineConfig,
    TrackingConfig,
    load_config,
)

__all__ = [
    "AppConfig",
    "DetectionConfig",
    "FaceConfig",
    "GimbalAxisConfig",
    "GimbalConfig",
    "IdentityConfig",
    "MotionPredictionConfig",
    "PROJECT_ROOT",
    "PerfMonitorConfig",
    "RecoveryConfig",
    "RegistrationConfig",
    "StateMachineConfig",
    "TrackingConfig",
    "load_config",
]
