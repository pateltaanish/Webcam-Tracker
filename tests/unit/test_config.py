"""Unit tests for webcam_tracker.config.settings."""

from __future__ import annotations

from webcam_tracker.config.settings import PROJECT_ROOT, AppConfig, load_config


def test_load_config_reads_yaml_defaults() -> None:
    config = load_config()
    assert config.video.source == "auto"
    assert config.video.requested_width == 1280
    assert config.video.requested_height == 720
    assert config.video.requested_fps == 30
    assert config.logging.level == "INFO"
    assert config.paths.models_dir == "models"
    assert config.detection.model_path == "yolo11n.pt"
    assert config.detection.confidence_threshold == 0.5
    assert config.detection.image_size == 640
    assert config.detection.device == "auto"
    assert config.tracking.lost_track_buffer == 30
    assert config.tracking.track_activation_threshold == 0.7
    assert config.tracking.minimum_consecutive_frames == 2
    assert config.tracking.minimum_iou_threshold == 0.1
    assert config.tracking.high_conf_det_threshold == 0.6
    assert config.perf_monitor.fps_window_seconds == 0.5
    assert config.perf_monitor.log_interval_seconds == 5.0
    assert config.gimbal.pan.kp == 80.0
    assert config.gimbal.pan.angle_limit_deg == 170.0
    assert config.gimbal.tilt.angle_limit_deg == 60.0
    assert config.gimbal.deadband == 0.02
    assert config.gimbal.max_velocity_deg_s == 120.0
    assert config.gimbal.max_acceleration_deg_s2 == 300.0
    assert config.gimbal.integral_limit == 20.0


def test_load_config_is_cached() -> None:
    assert load_config() is load_config()


def test_env_var_overrides_yaml_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("WEBCAM_TRACKER_VIDEO__SOURCE", "1")
    config = AppConfig()  # type: ignore[call-arg]
    assert config.video.source == "1"
    # Fields not overridden still come from the YAML default.
    assert config.video.requested_width == 1280


def test_resolve_path_is_relative_to_project_root() -> None:
    config = load_config()
    resolved = config.resolve_path(config.paths.models_dir)
    assert resolved == PROJECT_ROOT / "models"
    assert resolved.is_absolute()


def test_resolve_path_leaves_absolute_paths_unchanged() -> None:
    config = load_config()
    absolute = PROJECT_ROOT / "some" / "already" / "absolute" / "path"
    assert config.resolve_path(str(absolute)) == absolute
