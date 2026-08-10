"""Live MJPEG video + Server-Sent hazard events over plain HTTP.

Two independent channels off one server: GET /stream is MJPEG video, GET
/events is Server-Sent hazard events, GET / is a small built-in dashboard
wiring both together in a browser. This module only fans values out --
publish frames/events via LatestBroadcast from wherever the pipeline
already produces them (see scripts/preview_tracking.py).
"""

from webcam_tracker.streaming.server import LatestBroadcast, start_server

__all__ = ["LatestBroadcast", "start_server"]
