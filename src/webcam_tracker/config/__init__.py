"""Typed configuration loading (YAML defaults + environment variable overrides)."""

from webcam_tracker.config.settings import (
    PROJECT_ROOT,
    AppConfig,
    DetectionConfig,
    PerfMonitorConfig,
    TrackingConfig,
    load_config,
)

__all__ = [
    "AppConfig",
    "DetectionConfig",
    "PROJECT_ROOT",
    "PerfMonitorConfig",
    "TrackingConfig",
    "load_config",
]
