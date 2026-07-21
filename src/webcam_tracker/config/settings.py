"""Application configuration.

Values load from ``configs/default.yaml`` and can be overridden per-field by
environment variables (e.g. in a local ``.env`` file, gitignored) without
editing the YAML. Env vars always win over the YAML file, so machine-specific
or secret values never need to be committed.

Override convention: ``WEBCAM_TRACKER_<SECTION>__<FIELD>``, e.g.
``WEBCAM_TRACKER_VIDEO__SOURCE=1``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Repo root, computed from this file's location so nothing here depends on
# the current working directory a script happens to be launched from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class VideoConfig(BaseModel):
    """Camera / video-file input settings. Consumed by the video_input module."""

    source: str = Field(description="OpenCV device index as a string, or a video file path.")
    requested_width: int = Field(gt=0)
    requested_height: int = Field(gt=0)
    requested_fps: int = Field(gt=0)


class LoggingConfig(BaseModel):
    """Structured logging settings. Consumed by the logging_utils module."""

    level: str = "INFO"
    log_dir: str = "logs"
    json_format: bool = True


class PathsConfig(BaseModel):
    """Directories, given relative to the repo root unless already absolute."""

    models_dir: str = "models"
    data_dir: str = "data"


class DetectionConfig(BaseModel):
    """Person detector settings. Consumed by the detection module."""

    model_path: str = Field(description="Path to YOLO weights, relative to models_dir or absolute.")
    confidence_threshold: float = Field(gt=0.0, le=1.0)
    image_size: int = Field(gt=0, description="Model input size in pixels (square).")
    device: str = Field(description="'auto', 'cpu', 'cuda', or 'cuda:<index>'.")


class TrackingConfig(BaseModel):
    """Multi-object tracker (ByteTrack) settings. Consumed by the tracking module."""

    lost_track_buffer: int = Field(
        gt=0, description="Frames to keep a lost track alive for re-matching before dropping it."
    )
    track_activation_threshold: float = Field(
        gt=0.0, le=1.0, description="Min detection confidence to start a brand-new track."
    )
    minimum_consecutive_frames: int = Field(
        gt=0, description="Frames a new track must match before it's assigned a real (non -1) ID."
    )
    minimum_iou_threshold: float = Field(
        gt=0.0,
        le=1.0,
        description="Min box overlap (IoU) to match a detection to an existing track.",
    )
    high_conf_det_threshold: float = Field(
        gt=0.0,
        le=1.0,
        description="Confidence above which a detection is matched first (BYTE's 2-tier match).",
    )


class PerfMonitorConfig(BaseModel):
    """Performance monitoring settings. Consumed by the perf_monitor module."""

    fps_window_seconds: float = Field(
        gt=0.0, description="How often the rolling FPS figure is recomputed."
    )
    log_interval_seconds: float = Field(
        gt=0.0, description="Minimum time between structured perf-snapshot log lines."
    )


class AppConfig(BaseSettings):
    """Root application config, assembled from YAML defaults + env overrides."""

    model_config = SettingsConfigDict(
        yaml_file=str(os.environ.get("WEBCAM_TRACKER_CONFIG_FILE", _DEFAULT_CONFIG_PATH)),
        env_prefix="WEBCAM_TRACKER_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    video: VideoConfig
    logging: LoggingConfig
    paths: PathsConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    perf_monitor: PerfMonitorConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order = priority, highest first: explicit init args, then real env
        # vars, then a local .env file, then the committed YAML defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def resolve_path(self, relative_or_absolute: str) -> Path:
        """Resolve a config path against the repo root, unless already absolute."""
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load and cache the application config for the lifetime of the process."""
    return AppConfig()  # type: ignore[call-arg]  # fields populated by yaml/env sources
