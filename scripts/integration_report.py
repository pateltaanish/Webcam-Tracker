"""Stage 1.10 integration report: run the full pipeline over a real video and
print a structured, human-readable report of what the state machine did.

This is the tool for the qualitative Stage 1 exit check: record a few short
clips -- one person, two people crossing paths, a person leaving and
re-entering frame -- run this on each, and confirm the state timeline and
event counts make sense (target followed, losses escalate through occlusion
into a search, a returning person gets reacquired). It is not pass/fail
automation (that's tests/integration/test_pipeline.py); it's an eyes-on
summary of real footage.

Usage (from the repo root):
    # a single video file:
    .venv\\Scripts\\python.exe scripts\\integration_report.py data\\clips\\cross.mp4
    # a folder of clips, played back to back:
    .venv\\Scripts\\python.exe scripts\\integration_report.py data\\clips
    # no path -> uses configs/default.yaml's source ("auto" webcam), capped:
    .venv\\Scripts\\python.exe scripts\\integration_report.py

Optional second arg caps the number of frames (default 600, mainly so a live
webcam run terminates):
    .venv\\Scripts\\python.exe scripts\\integration_report.py data\\clips\\a.mp4 300

State transitions are timed off VIDEO time (frame index / configured fps), not
wall clock, so the occlusion/give-up timings in the report are faithful no
matter how fast inference runs.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

# A video-path argument overrides the configured source. Must be set before
# load_config() (config is cached on first load).
_MAX_FRAMES_DEFAULT = 600
if len(sys.argv) > 1:
    os.environ["WEBCAM_TRACKER_VIDEO__SOURCE"] = sys.argv[1]
_max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else _MAX_FRAMES_DEFAULT

from webcam_tracker.config import load_config  # noqa: E402 -- must follow the env override above
from webcam_tracker.detection import create_detector  # noqa: E402
from webcam_tracker.logging_utils import configure_logging  # noqa: E402
from webcam_tracker.state_machine import (  # noqa: E402
    SystemStatus,
    TrackingState,
    create_state_machine,
)
from webcam_tracker.tracking import create_tracker  # noqa: E402
from webcam_tracker.video_input import VideoSourceError, create_source  # noqa: E402


class _VideoClock:
    """Advances by one frame-period each frame, so time reflects VIDEO
    position rather than how fast inference happens to run."""

    def __init__(self, fps: float) -> None:
        self._t = 0.0
        self._dt = 1.0 / fps

    def __call__(self) -> float:
        return self._t

    def tick(self) -> None:
        self._t += self._dt


def _print_report(source_desc: str, fps: float, timeline: list[SystemStatus]) -> None:
    print("\n" + "=" * 64)
    print(f"Integration report: {source_desc}")
    print("=" * 64)
    if not timeline:
        print("No frames processed.")
        return

    frames = len(timeline)
    print(f"frames processed: {frames}  (video time {frames / fps:.1f}s @ {fps:.0f}fps)")

    print("\nstate distribution:")
    counts = Counter(s.state for s in timeline)
    for state in TrackingState:
        n = counts.get(state, 0)
        if n:
            print(f"  {state.value:22} {n:5}  ({100 * n / frames:4.0f}%)")

    # Transitions (with video time), and event tallies derived from them.
    transitions: list[tuple[float, TrackingState, TrackingState]] = []
    losses = searches = give_ups = reacquisitions = 0
    prev = timeline[0].state
    for i, status in enumerate(timeline[1:], start=1):
        if status.state != prev:
            transitions.append((i / fps, prev, status.state))
            if status.state is TrackingState.TEMPORARILY_OCCLUDED:
                losses += 1
            elif status.state is TrackingState.RECOVERY_SEARCH:
                searches += 1
            elif status.state is TrackingState.SAFE_HOVER_REQUESTED:
                give_ups += 1
            prev = status.state
        if status.reacquired_track_id is not None:
            reacquisitions += 1

    print(f"\nstate transitions: {len(transitions)}")
    for t, a, b in transitions[:40]:
        print(f"  t={t:6.2f}s  {a.value} -> {b.value}")
    if len(transitions) > 40:
        print(f"  ... ({len(transitions) - 40} more)")

    print("\nevent summary:")
    print(f"  target losses (-> temporarily_occluded): {losses}")
    print(f"  escalated to recovery search:            {searches}")
    print(f"  reacquisitions (re-locked a new track):  {reacquisitions}")
    print(f"  gave up (-> safe_hover_requested):       {give_ups}")

    tracked_fraction = 100 * counts.get(TrackingState.TRACKING, 0) / frames
    print(
        f"\nverdict: target was actively tracked {tracked_fraction:.0f}% of frames; "
        f"{losses} loss(es), {reacquisitions} reacquisition(s). "
        "Check that this matches what happens in the clip."
    )


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)
    fps = float(config.video.requested_fps)
    clock = _VideoClock(fps)
    state_machine = create_state_machine(config, clock=clock)

    source = create_source(config)
    source_desc = config.video.source

    timeline: list[SystemStatus] = []
    selected = False
    try:
        with source:
            for frame in source:
                clock.tick()
                tracked = tracker.update(detector.detect(frame.image))
                if not selected and tracked:
                    state_machine.selector.select(tracked[0].track_id)
                    selected = True
                    print(f"auto-selected first confirmed track_id={tracked[0].track_id}")
                height, width = frame.image.shape[:2]
                timeline.append(state_machine.update(tracked, width, height))
                if len(timeline) >= _max_frames:
                    print(f"(reached frame cap {_max_frames})")
                    break
    except VideoSourceError as exc:
        raise SystemExit(f"Could not open video source '{source_desc}': {exc}") from exc
    except KeyboardInterrupt:
        pass

    if not selected:
        print("No track was ever confirmed -- nothing to report. Is anyone in the clip?")
    _print_report(source_desc, fps, timeline)


if __name__ == "__main__":
    main()
