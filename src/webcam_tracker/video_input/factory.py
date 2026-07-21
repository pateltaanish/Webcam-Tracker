"""Picks the right FrameSource implementation from config.

This is the one place that interprets `config.video.source` as either the
literal string "auto", a webcam device index, a file path, or a folder path
— everywhere else in the codebase just gets a FrameSource and doesn't care
which kind it is.
"""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.video_input.base import FrameSource, VideoSourceError
from webcam_tracker.video_input.file_source import VideoFileSource
from webcam_tracker.video_input.folder_source import FolderSource
from webcam_tracker.video_input.webcam_source import WebcamSource


def create_source(config: AppConfig) -> FrameSource:
    """Build a FrameSource from `config.video`. Does not call open() — the
    caller controls the source's lifetime (typically via a `with` block)."""
    source = config.video.source.strip()

    if source.lower() == "auto":
        return WebcamSource(
            device_index=None,
            requested_width=config.video.requested_width,
            requested_height=config.video.requested_height,
            requested_fps=config.video.requested_fps,
        )

    if source.isdigit():
        return WebcamSource(
            device_index=int(source),
            requested_width=config.video.requested_width,
            requested_height=config.video.requested_height,
            requested_fps=config.video.requested_fps,
        )

    path = config.resolve_path(source)
    if path.is_dir():
        return FolderSource(path)
    if path.is_file():
        return VideoFileSource(path)

    raise VideoSourceError(
        f"video.source '{source}' is not 'auto', a webcam index, or an existing "
        f"file/directory (resolved to {path}). Check configs/default.yaml or your "
        "WEBCAM_TRACKER_VIDEO__SOURCE override."
    )
