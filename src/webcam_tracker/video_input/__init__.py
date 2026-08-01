"""Video input.

Abstracts webcam, video-file, folder-of-videos, and (later) network camera
sources behind a single FrameSource interface, so every downstream module is
agnostic to where frames come from.
"""

from webcam_tracker.video_input.base import FrameSource, VideoFrame, VideoSourceError
from webcam_tracker.video_input.factory import create_source
from webcam_tracker.video_input.file_source import VideoFileSource
from webcam_tracker.video_input.folder_source import FolderSource
from webcam_tracker.video_input.picamera_source import PiCameraSource
from webcam_tracker.video_input.webcam_source import (
    BACKEND_NAMES,
    CameraProbeResult,
    WebcamSource,
    discover_webcam_index,
    probe_cameras,
)

__all__ = [
    "FrameSource",
    "VideoFrame",
    "VideoSourceError",
    "create_source",
    "VideoFileSource",
    "FolderSource",
    "PiCameraSource",
    "WebcamSource",
    "discover_webcam_index",
    "probe_cameras",
    "CameraProbeResult",
    "BACKEND_NAMES",
]
