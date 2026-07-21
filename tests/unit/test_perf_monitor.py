"""Unit tests for webcam_tracker.perf_monitor.

Uses a fake, manually-advanced clock so FPS/latency math is deterministic --
no real sleeping, no timing flakiness.
"""

from __future__ import annotations

import pytest

from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.perf_monitor import monitor as monitor_module


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestPerfMonitor:
    def test_frame_count_increments_each_start_frame(self) -> None:
        clock = FakeClock()
        monitor = PerfMonitor(clock=clock)
        monitor.start_frame()
        monitor.start_frame()
        monitor.start_frame()
        assert monitor.snapshot().frame_count == 3

    def test_fps_updates_only_after_window_elapses(self) -> None:
        clock = FakeClock()
        monitor = PerfMonitor(fps_window_seconds=1.0, clock=clock)

        clock.advance(0.3)
        monitor.start_frame()
        assert monitor.snapshot().fps == 0.0  # window not elapsed yet

        clock.advance(0.3)
        monitor.start_frame()
        assert monitor.snapshot().fps == 0.0  # still not elapsed (0.6s total)

        clock.advance(0.5)  # total 1.1s, 3 frames
        monitor.start_frame()
        assert monitor.snapshot().fps == pytest.approx(3 / 1.1)

    def test_measure_records_stage_latency_ms(self) -> None:
        clock = FakeClock()
        monitor = PerfMonitor(clock=clock)

        with monitor.measure("detection"):
            clock.advance(0.025)

        assert monitor.snapshot().stage_latency_ms["detection"] == pytest.approx(25.0)

    def test_multiple_stages_recorded_independently(self) -> None:
        clock = FakeClock()
        monitor = PerfMonitor(clock=clock)

        with monitor.measure("detection"):
            clock.advance(0.010)
        with monitor.measure("tracking"):
            clock.advance(0.005)

        latencies = monitor.snapshot().stage_latency_ms
        assert latencies["detection"] == pytest.approx(10.0)
        assert latencies["tracking"] == pytest.approx(5.0)

    def test_end_frame_records_total_latency(self) -> None:
        clock = FakeClock()
        monitor = PerfMonitor(clock=clock)

        monitor.start_frame()
        clock.advance(0.05)
        monitor.end_frame()

        assert monitor.snapshot().total_frame_latency_ms == pytest.approx(50.0)

    def test_snapshot_is_a_detached_copy(self) -> None:
        clock = FakeClock()
        monitor = PerfMonitor(clock=clock)
        with monitor.measure("a"):
            clock.advance(0.01)
        first_snapshot = monitor.snapshot()

        with monitor.measure("b"):
            clock.advance(0.01)

        assert "b" not in first_snapshot.stage_latency_ms

    def test_log_if_due_respects_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []
        monkeypatch.setattr(
            monitor_module.logger, "info", lambda msg, extra=None: calls.append(extra)
        )

        clock = FakeClock()
        monitor = PerfMonitor(clock=clock)

        monitor.start_frame()
        monitor.log_if_due(interval_seconds=1.0)
        monitor.log_if_due(interval_seconds=1.0)  # too soon, must not log again
        assert len(calls) == 1

        clock.advance(1.5)
        monitor.log_if_due(interval_seconds=1.0)
        assert len(calls) == 2
