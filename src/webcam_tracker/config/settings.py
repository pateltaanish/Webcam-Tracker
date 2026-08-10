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

from pydantic import BaseModel, Field, model_validator
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
    record: bool = Field(
        default=False,
        description="Write each preview script's annotated output to data/recordings/.",
    )


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


class GimbalAxisConfig(BaseModel):
    """PID gains + angle limit for one gimbal axis (pan or tilt)."""

    kp: float = Field(description="Proportional gain: deg/s of output per unit normalized error.")
    ki: float = Field(ge=0.0, description="Integral gain: corrects small persistent error.")
    kd: float = Field(ge=0.0, description="Derivative gain: dampens oscillation/overshoot.")
    angle_limit_deg: float = Field(
        gt=0.0, description="Max angle from neutral (0) this axis can reach."
    )


class GimbalConfig(BaseModel):
    """Simulated gimbal controller settings (Stage 1.7). No real motors are
    ever driven by this config -- see src/webcam_tracker/gimbal_control.

    Gain values below are untuned placeholders (no physical gimbal exists
    yet to tune against) -- chosen to produce plausible-looking simulated
    motion, not validated against real hardware dynamics. Revisit once
    Stage 3 has real hardware, or sooner if the live preview looks off.
    """

    pan: GimbalAxisConfig
    tilt: GimbalAxisConfig
    deadband: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized error (0-1) below which output is forced to zero, to avoid jitter.",
    )
    max_velocity_deg_s: float = Field(
        gt=0.0, description="Max commanded angular velocity, either axis."
    )
    max_acceleration_deg_s2: float = Field(
        gt=0.0, description="Max change in commanded velocity per second (rate limiting)."
    )
    integral_limit: float = Field(
        gt=0.0, description="Clamp on the accumulated integral term (anti-windup)."
    )


class MotionPredictionConfig(BaseModel):
    """Per-track constant-velocity Kalman filter settings (Stage 1.8).

    Operates in pixel coordinates (that's what tracked boxes are in). Consumed
    by the motion_prediction module. Values are starting points tuned to look
    reasonable on real webcam tracks, not derived from a calibrated noise
    model -- adjust if predicted positions lag or jitter.
    """

    process_noise: float = Field(
        gt=0.0,
        description="Assumed acceleration variance: higher = filter follows sudden "
        "velocity changes faster but is jerkier/noisier.",
    )
    measurement_noise: float = Field(
        gt=0.0,
        description="Assumed detection-position noise: higher = smoother but laggier "
        "estimates (trusts the model over each new measurement).",
    )
    max_coast_seconds: float = Field(
        gt=0.0,
        description="How long to keep a track's filter alive with no new measurement "
        "before discarding it -- lets recovery read a just-lost target's last-known "
        "velocity. Independent of the tracker's own lost_track_buffer.",
    )


class RecoveryConfig(BaseModel):
    """Target-loss search + reacquisition settings (Stage 1.8). Consumed by the
    recovery module.

    NOTE: Stage 1 reacquisition is identity-FREE -- it re-locks onto the
    nearest newly-appearing track near the predicted position, whoever that
    is. It is a deliberately-labeled placeholder for the real identity-gated
    reacquisition that arrives in Stage 2 (face/re-id). These knobs tune the
    *mechanics* only.
    """

    give_up_seconds: float = Field(
        gt=0.0, description="How long to search after a loss before giving up on the target."
    )
    prediction_horizon_seconds: float = Field(
        gt=0.0,
        description="Cap on how far ahead the last-known velocity is extrapolated to "
        "estimate where the target went -- a stale velocity extrapolated too long "
        "would fly off-screen.",
    )
    reacquire_radius_fraction: float = Field(
        gt=0.0,
        le=2.0,
        description="A newly-appearing track re-locks only if within this fraction of "
        "the frame diagonal from the predicted target position. Larger = more eager "
        "(and more likely to grab the wrong person, since there's no identity check yet).",
    )
    search_amplitude: float = Field(
        gt=0.0,
        le=1.0,
        description="Normalized (0-1) amplitude of the simulated search sweep fed to the "
        "gimbal while searching -- how far off-center the search points.",
    )
    search_period_seconds: float = Field(
        gt=0.0, description="Duration of one full back-and-forth search sweep oscillation."
    )


class HazardConfig(BaseModel):
    """Looming-based hazard detection. Consumed by the hazard module.

    CALIBRATION WARNING: every default in configs/default.yaml is inherited
    from the Watchdog wearable, where it was tuned against 176 real fired
    alerts on chest-height footage of someone WALKING. Two of them encode
    that body and that speed directly -- static_bottom_y assumes a
    forward-facing camera at chest height, ttc_alert assumes walking
    closure rates -- and neither transfers to a camera on a drone. Re-derive
    them from drone footage before trusting an alert; the code ports, the
    numbers do not.
    """

    min_samples: int = Field(
        ge=2,
        description="Frames of history before a TTC estimate is attempted at all. "
        "This is the alert latency floor: at 9 FPS on the Pi's CPU, 5 samples is "
        "~0.54s before a closing person can register, vs ~0.21s at Watchdog's 24 "
        "FPS. Lower reacts sooner on a noisier fit.",
    )
    sample_window: int = Field(
        ge=2,
        description="Box-width samples kept per track and fed to the growth-rate "
        "fit. Longer smooths noise but spans more real time, over which the "
        "straight-line fit to a curved (~1/distance) growth carries more error.",
    )
    min_width_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Boxes narrower than this fraction of frame width get no TTC -- "
        "too few pixels for the growth rate to be measurable rather than noise.",
    )
    ttc_min: float = Field(
        gt=0.0, description="Floor on the reported TTC, clamping implausibly fast closures."
    )
    ttc_max: float = Field(
        gt=0.0, description="Ceiling on the reported TTC, clamping barely-growing boxes."
    )
    zone_discard_margin: float = Field(
        ge=0.0,
        lt=0.5,
        description="People whose box center is within this fraction of either frame "
        "edge are discarded entirely: lens distortion there makes both position and "
        "width (hence TTC) untrustworthy, and they're usually half out of frame.",
    )
    zone_left_max: float = Field(
        gt=0.0, lt=1.0, description="Box centers below this normalized x report zone 'left'."
    )
    zone_right_min: float = Field(
        gt=0.0, lt=1.0, description="Box centers above this normalized x report zone 'right'."
    )
    ttc_alert: float = Field(
        gt=0.0,
        description="A closing person only becomes a hazard at or below this TTC. "
        "This is the gate on whether anything is reported at all.",
    )
    ttc_urgent: float = Field(
        gt=0.0,
        description="Hazards below this TTC are flagged urgent. Only labels a hazard "
        "that is already firing -- it never suppresses one -- so it's safe to retune "
        "without risking silence.",
    )
    track_cooldown: float = Field(
        gt=0.0, description="Minimum gap before the SAME track can be reported again."
    )
    static_cooldown: float = Field(
        gt=0.0,
        description="Longer cooldown for close-but-not-closing hazards. A stationary "
        "obstacle stays in frame indefinitely, so it would otherwise re-fire forever.",
    )
    global_gap: float = Field(
        gt=0.0,
        description="Minimum gap between ANY two hazard reports, across all tracks -- "
        "stops a crowd from producing a continuous stream of them.",
    )
    static_bottom_y: float = Field(
        gt=0.0,
        le=1.0,
        description="A person with no measurable looming still counts as a hazard once "
        "their box bottom reaches this normalized y (close enough to fill the bottom of "
        "frame). Encodes camera height and pitch -- see the class-level warning.",
    )

    @model_validator(mode="after")
    def _check_ordering(self) -> HazardConfig:
        """Catch orderings that disable a feature silently rather than erroring."""
        if self.sample_window < self.min_samples:
            raise ValueError(
                f"sample_window ({self.sample_window}) < min_samples ({self.min_samples}): "
                "the history is capped below the minimum, so TTC could never be computed."
            )
        if self.ttc_min >= self.ttc_max:
            raise ValueError(f"ttc_min ({self.ttc_min}) must be < ttc_max ({self.ttc_max})")
        if self.zone_left_max > self.zone_right_min:
            raise ValueError(
                f"zone_left_max ({self.zone_left_max}) must be <= zone_right_min "
                f"({self.zone_right_min}), else 'center' is empty and no box can report it."
            )
        if self.ttc_urgent > self.ttc_alert:
            raise ValueError(
                f"ttc_urgent ({self.ttc_urgent}) must be <= ttc_alert ({self.ttc_alert}), "
                "else every reported hazard is urgent and the distinction is dead."
            )
        return self


class StateMachineConfig(BaseModel):
    """Tracking state machine settings (Stage 1.9). Consumed by the
    state_machine module, which coordinates selection + prediction + recovery
    + gimbal into one authoritative per-frame decision.
    """

    occlusion_timeout_seconds: float = Field(
        gt=0.0,
        description="How long the target can be missing before a brief dropout "
        "(TEMPORARILY_OCCLUDED -- hold position, wait for the same track to "
        "return) escalates into an active recovery search (RECOVERY_SEARCH). "
        "Roughly match the tracker's lost_track_buffer duration: within that "
        "window the tracker usually re-emits the SAME track id, so holding beats "
        "swinging the camera or grabbing a different person.",
    )


class IdentityConfig(BaseModel):
    """Identity / encrypted-profile-database settings (Stage 2). Consumed by
    the database module. See docs/database_design.md.

    The encryption key is NEVER stored here or anywhere on disk -- it's derived
    at runtime from the operator passphrase. These are only the store location
    and the (non-secret) KDF cost parameters.
    """

    store_dir: str = Field(
        description="Directory (relative to repo root unless absolute) holding the "
        "encrypted profile database and its key-vault sidecar. Gitignored."
    )
    db_filename: str = Field(description="SQLite profile-store filename within store_dir.")
    keyvault_filename: str = Field(
        description="Plaintext key-vault sidecar (salt + KDF params + wrapped data key) "
        "within store_dir. Safe to store in the clear; useless without the passphrase."
    )
    accounts_filename: str = Field(
        default="accounts.json",
        description="Multi-user login file (Stage 2.5) within store_dir: mode, KDF "
        "params, per-store name-hash salt, and wrapped data keys. Safe in the clear; "
        "holds no passphrases or raw names.",
    )
    argon2_time_cost: int = Field(
        gt=0, description="Argon2id iterations. Higher = slower unlock, harder brute force."
    )
    argon2_memory_kib: int = Field(
        ge=8192,
        description="Argon2id memory in KiB. Higher = more GPU-brute-force-resistant "
        "(and more RAM per unlock).",
    )
    argon2_parallelism: int = Field(gt=0, description="Argon2id parallelism (lanes).")
    min_passphrase_length: int = Field(
        gt=0, description="Minimum operator passphrase length enforced at setup."
    )


class FaceConfig(BaseModel):
    """Face detection + embedding model settings (Stage 2.2). Consumed by the
    face_recognition module. See docs/model_licenses.md for the InsightFace
    buffalo_l non-commercial license caveat."""

    model_pack: str = Field(
        description="InsightFace model pack name, e.g. 'buffalo_l' (SCRFD + ArcFace)."
    )
    det_size: int = Field(gt=0, description="Square detector input size in pixels (e.g. 640).")
    device: str = Field(
        description="'cpu' (onnxruntime) or 'cuda' (needs onnxruntime-gpu + matching CUDA)."
    )
    match_threshold: float = Field(
        gt=0.0,
        le=1.0,
        description="Minimum cosine similarity (ArcFace embeddings are unit vectors, so "
        "similarity is in -1..1) for a face to count as a registered person; below this "
        "the result is UNKNOWN. Starting placeholder -- tune on real data in Stage 2.5 "
        "(FAR/FRR). Higher = fewer false accepts, more false rejects.",
    )


class RegistrationConfig(BaseModel):
    """Registration / enrollment settings (Stage 2.2). Consumed by the
    registration module: how many samples to require and the quality gates a
    captured face must pass before it's embedded and stored."""

    consent_version: str = Field(
        description="Consent-terms version recorded with each enrollment (bump if terms change)."
    )
    samples_required: int = Field(
        gt=0, description="Number of quality-passing face samples needed to enroll a person."
    )
    min_detection_score: float = Field(
        gt=0.0, le=1.0, description="Minimum face detector confidence to accept a sample."
    )
    min_blur_variance: float = Field(
        gt=0.0,
        description="Minimum variance-of-Laplacian on the face crop (sharpness gate); "
        "below this the face is too blurry. Resolution-dependent -- tune per camera.",
    )
    min_brightness: float = Field(
        ge=0.0, le=255.0, description="Minimum mean face-crop brightness (rejects too-dark)."
    )
    max_brightness: float = Field(
        ge=0.0, le=255.0, description="Maximum mean face-crop brightness (rejects blown-out)."
    )
    min_face_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Minimum face-box area as a fraction of the frame (person must be "
        "close enough for a reliable embedding).",
    )


class IdentityTrackingConfig(BaseModel):
    """Per-track identity fusion settings (Stage 2.4). Consumed by the identity
    module, which assigns registered-person identities to live tracks using
    face matches + temporal consistency.

    Running the face model every frame is too slow, so identity is refreshed
    periodically; between refreshes each track keeps its last identity.
    """

    update_every_n_frames: int = Field(
        gt=0,
        description="Run the face model + matching once every N frames (tracking still "
        "runs every frame). Higher = cheaper but slower to confirm/switch identity.",
    )
    history_window: int = Field(
        gt=0,
        description="How many recent identity refreshes to remember per track. The "
        "consensus over this window is the track's identity (filters single-frame "
        "mismatches).",
    )
    min_confidence: float = Field(
        gt=0.0,
        le=1.0,
        description="Fraction of the history window that must agree on a person before a "
        "track is confirmed as that person (else the track is unconfirmed/unknown). This "
        "is what makes reacquisition identity-GATED, not geometric.",
    )


class AppConfig(BaseSettings):
    """Root application config, assembled from YAML defaults + env overrides."""

    model_config = SettingsConfigDict(
        yaml_file=str(os.environ.get("WEBCAM_TRACKER_CONFIG_FILE", _DEFAULT_CONFIG_PATH)),
        # Absolute, not ".env" -- pydantic-settings resolves a relative env_file
        # against the *current working directory*, so scripts run from anywhere
        # but the repo root would silently ignore it. Without this line the
        # dotenv source below has no file at all and .env is a no-op.
        env_file=PROJECT_ROOT / ".env",
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
    gimbal: GimbalConfig
    motion_prediction: MotionPredictionConfig
    recovery: RecoveryConfig
    hazard: HazardConfig
    state_machine: StateMachineConfig
    identity: IdentityConfig
    face: FaceConfig
    registration: RegistrationConfig
    identity_tracking: IdentityTrackingConfig

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
