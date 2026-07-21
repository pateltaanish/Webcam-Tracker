"""Typed configuration loading (YAML defaults + environment variable overrides)."""

from webcam_tracker.config.settings import PROJECT_ROOT, AppConfig, DetectionConfig, load_config

__all__ = ["AppConfig", "DetectionConfig", "PROJECT_ROOT", "load_config"]
