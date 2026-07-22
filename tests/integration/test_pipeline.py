"""Stage 1.10 integration pass: the WHOLE Stage 1 pipeline end to end.

Runs the real detector + real tracker + real state machine (with all its real
sub-components) over short synthetic "videos" built from Ultralytics' bundled
sample images -- real inference on real people, but with frame contents we
control, so we can script the three roadmap scenarios (single person,
leave/re-enter, multiple people) and assert the *mechanics* behave: a target
gets tracked and followed, a disappearance escalates through occlusion into a
recovery search, and a person returning as a NEW track id gets reacquired.

Timing is driven by an injected fake clock (advanced a fixed dt per frame) so
the time-based state transitions (occlusion timeout, recovery give-up) are
deterministic regardless of how fast the test host runs inference. The
tracker's own confirmation/lost-track buffers are frame-based, so they're
deterministic too.

This is the Stage 1 exit criterion: not perfect tracking, just correct
mechanics we can build identity (Stage 2) on top of.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics

from webcam_tracker.config import load_config
from webcam_tracker.detection import PersonDetector, create_detector
from webcam_tracker.state_machine import SystemStatus, TrackingState, TrackingStateMachine
from webcam_tracker.state_machine import create_state_machine as _create_state_machine
from webcam_tracker.tracking import TrackedPerson, create_tracker

_ASSETS_DIR = Path(ultralytics.__file__).resolve().parent / "assets"


class _FakeClock:
    """Manually-advanced clock (mirrors tests/unit/_fake_clock.py; kept local
    to avoid a cross-package test import)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(scope="module")
def detector() -> PersonDetector:
    config = load_config()
    det = create_detector(config)
    det.load()
    return det


def _load(name: str) -> np.ndarray:
    image = cv2.imread(str(_ASSETS_DIR / name))
    assert image is not None, f"bundled sample {name} failed to load"
    return image


def _blank_like(image: np.ndarray) -> np.ndarray:
    """A same-size frame that yields zero detections (uniform gray)."""
    return np.full_like(image, 128)


def _run(
    detector: PersonDetector,
    frames: list[np.ndarray],
    *,
    dt: float = 0.1,
) -> tuple[list[SystemStatus], TrackingStateMachine, list[list[TrackedPerson]]]:
    """Drive the full pipeline over `frames`, auto-selecting the first confirmed
    track. Returns the per-frame system status, the state machine, and the
    tracked people seen each frame."""
    config = load_config()
    clock = _FakeClock()
    tracker = create_tracker(config)
    state_machine = _create_state_machine(config, clock=clock)

    timeline: list[SystemStatus] = []
    tracked_per_frame: list[list[TrackedPerson]] = []
    selected = False
    for image in frames:
        clock.advance(dt)
        tracked = tracker.update(detector.detect(image))
        tracked_per_frame.append(tracked)
        if not selected and tracked:
            state_machine.selector.select(tracked[0].track_id)
            selected = True
        height, width = image.shape[:2]
        timeline.append(state_machine.update(tracked, width, height))
    return timeline, state_machine, tracked_per_frame


class TestFullPipelineIntegration:
    def test_single_person_tracked_and_followed_stably(self, detector: PersonDetector) -> None:
        timeline, _, _ = _run(detector, [_load("zidane.jpg")] * 8)
        states = [s.state for s in timeline]

        assert TrackingState.TRACKING in states
        assert timeline[-1].state is TrackingState.TRACKING
        # The selected target's id never switches while continuously visible.
        target_ids = {
            s.target_status.target_id
            for s in timeline
            if s.state is TrackingState.TRACKING and s.target_status is not None
        }
        assert len(target_ids) == 1

    def test_multiple_people_both_tracked_and_one_selected(self, detector: PersonDetector) -> None:
        timeline, state_machine, tracked_per_frame = _run(detector, [_load("bus.jpg")] * 8)

        # The pipeline handles more than one person at once.
        assert max(len(t) for t in tracked_per_frame) >= 2
        # ...while the state machine locks onto and follows exactly one.
        assert timeline[-1].state is TrackingState.TRACKING
        assert state_machine.selector.target_id is not None

    def test_leave_and_reenter_recovers(self, detector: PersonDetector) -> None:
        person = _load("zidane.jpg")
        blank = _blank_like(person)
        # 15 blank frames * 0.1s = 1.5s > occlusion timeout (1.0s) -> escalates
        # to a search; 15 frames < lost_track_buffer (30) -> the SAME id
        # resumes when the person returns.
        frames = [person] * 6 + [blank] * 15 + [person] * 8
        timeline, _, _ = _run(detector, frames)
        states = [s.state for s in timeline]

        assert TrackingState.TRACKING in states
        assert TrackingState.TEMPORARILY_OCCLUDED in states
        assert TrackingState.RECOVERY_SEARCH in states
        assert timeline[-1].state is TrackingState.TRACKING  # recovered

    def test_reenter_as_new_id_is_reacquired(self, detector: PersonDetector) -> None:
        person = _load("zidane.jpg")
        blank = _blank_like(person)
        # 35 blank frames > lost_track_buffer (30): the returning person is a
        # brand-NEW track id -- exactly the "leaves and comes back as a new ID"
        # case. Still within recovery's give-up window (5.0s), so recovery
        # should re-lock onto the new track near the predicted position.
        frames = [person] * 6 + [blank] * 35 + [person] * 10
        timeline, state_machine, _ = _run(detector, frames)
        states = [s.state for s in timeline]

        initial_target = next(
            s.target_status.target_id
            for s in timeline
            if s.state is TrackingState.TRACKING and s.target_status is not None
        )
        reacquired = [s.reacquired_track_id for s in timeline if s.reacquired_track_id is not None]

        assert TrackingState.RECOVERY_SEARCH in states
        assert reacquired, "expected recovery to reacquire the returning person as a new track"
        assert timeline[-1].state is TrackingState.TRACKING
        # The reacquired id is a genuinely new track, not the original one.
        assert state_machine.selector.target_id != initial_target
