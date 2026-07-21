"""Builds a PersonDetector from config, resolving the model path against
paths.models_dir the same way every other module resolves config paths."""

from __future__ import annotations

from pathlib import Path

from webcam_tracker.config import AppConfig
from webcam_tracker.detection.detector import PersonDetector


def create_detector(config: AppConfig) -> PersonDetector:
    models_dir = config.resolve_path(config.paths.models_dir)
    model_path = Path(config.detection.model_path)
    if not model_path.is_absolute():
        model_path = models_dir / model_path

    return PersonDetector(
        model_path=model_path,
        confidence_threshold=config.detection.confidence_threshold,
        image_size=config.detection.image_size,
        device=config.detection.device,
    )
