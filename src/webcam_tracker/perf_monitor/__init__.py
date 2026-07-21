"""Performance monitoring.

Tracks FPS and per-stage processing latency (e.g. detection vs. tracking
time within a frame). CPU/GPU/RAM/thermal tracking is added in Stage 3.
"""

from webcam_tracker.perf_monitor.monitor import PerfMonitor, PerfSnapshot

__all__ = ["PerfMonitor", "PerfSnapshot"]
