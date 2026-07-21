"""Frame-rate and per-stage latency tracking.

This replaces the FPS-counter code that used to be duplicated inline in
every preview script (`scripts/preview_*.py`) with one tested
implementation, and adds per-stage latency (how long detection vs. tracking
took within a frame) that the scripts didn't measure at all before.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerfSnapshot:
    """A point-in-time read of the monitor's current state."""

    fps: float
    frame_count: int
    total_frame_latency_ms: float
    stage_latency_ms: dict[str, float] = field(default_factory=dict)


class PerfMonitor:
    """Tracks rolling FPS and per-stage latency across a frame-processing loop.

    Usage::

        monitor = PerfMonitor(fps_window_seconds=0.5)
        for frame in source:
            monitor.start_frame()
            with monitor.measure("detection"):
                detections = detector.detect(frame.image)
            with monitor.measure("tracking"):
                tracked = tracker.update(detections)
            monitor.end_frame()
            snapshot = monitor.snapshot()  # .fps, .stage_latency_ms, ...
    """

    def __init__(
        self,
        fps_window_seconds: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`clock` defaults to time.monotonic; tests inject a fake clock for
        deterministic FPS calculations without real sleeping."""
        self._fps_window_seconds = fps_window_seconds
        self._clock = clock

        self._window_start = clock()
        self._frames_in_window = 0
        self._fps = 0.0
        self._frame_count = 0

        self._frame_start: float | None = None
        self._total_frame_latency_ms = 0.0
        self._stage_latency_ms: dict[str, float] = {}
        # -inf (not 0.0) so the first log_if_due() call always fires,
        # regardless of what value the clock happens to start at.
        self._last_log_time = float("-inf")

    def start_frame(self) -> None:
        """Call once at the top of each loop iteration."""
        now = self._clock()
        self._frame_start = now
        self._frame_count += 1
        self._frames_in_window += 1

        elapsed = now - self._window_start
        if elapsed >= self._fps_window_seconds:
            self._fps = self._frames_in_window / elapsed
            self._frames_in_window = 0
            self._window_start = now

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        """Time a named stage (e.g. "detection", "tracking") within the current frame."""
        start = self._clock()
        try:
            yield
        finally:
            self._stage_latency_ms[stage_name] = (self._clock() - start) * 1000

    def end_frame(self) -> None:
        """Call once at the bottom of each loop iteration, after all `measure()` blocks."""
        if self._frame_start is not None:
            self._total_frame_latency_ms = (self._clock() - self._frame_start) * 1000

    def snapshot(self) -> PerfSnapshot:
        """A read-only copy of the current FPS/latency state."""
        return PerfSnapshot(
            fps=self._fps,
            frame_count=self._frame_count,
            total_frame_latency_ms=self._total_frame_latency_ms,
            stage_latency_ms=dict(self._stage_latency_ms),
        )

    def log_if_due(self, interval_seconds: float) -> None:
        """Log a structured perf snapshot, at most once per `interval_seconds`.

        Call this every frame; it no-ops most calls and only actually logs
        when the interval has elapsed, so continuous per-frame monitoring
        doesn't spam the log file (spec requirement: FPS/latency go to
        structured logs, without flooding them).
        """
        now = self._clock()
        if now - self._last_log_time >= interval_seconds:
            snapshot = self.snapshot()
            logger.info(
                "Perf snapshot",
                extra={
                    "fps": round(snapshot.fps, 1),
                    "frame_count": snapshot.frame_count,
                    "total_frame_latency_ms": round(snapshot.total_frame_latency_ms, 1),
                    "stage_latency_ms": {
                        k: round(v, 1) for k, v in snapshot.stage_latency_ms.items()
                    },
                },
            )
            self._last_log_time = now
