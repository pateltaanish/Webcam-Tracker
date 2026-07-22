"""A manually-advanced fake clock, shared by tests for anything that takes an
injectable `Callable[[], float]` clock (PerfMonitor, AxisController, ...).
Not a test module itself -- doesn't match pytest's test-file naming pattern.
"""

from __future__ import annotations


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
