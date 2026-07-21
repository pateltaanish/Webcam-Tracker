"""Builds a PersonTracker from config."""

from __future__ import annotations

from webcam_tracker.config import AppConfig
from webcam_tracker.tracking.tracker import PersonTracker


def create_tracker(config: AppConfig) -> PersonTracker:
    # frame_rate feeds ByteTrack's internal buffer-frames<->time conversion.
    # We use the *requested* fps rather than a live-negotiated value since
    # the tracker is constructed before a video source is opened.
    return PersonTracker(
        lost_track_buffer=config.tracking.lost_track_buffer,
        frame_rate=float(config.video.requested_fps),
        track_activation_threshold=config.tracking.track_activation_threshold,
        minimum_consecutive_frames=config.tracking.minimum_consecutive_frames,
        minimum_iou_threshold=config.tracking.minimum_iou_threshold,
        high_conf_det_threshold=config.tracking.high_conf_det_threshold,
    )
