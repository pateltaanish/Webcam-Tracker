"""Picks at most one hazard per frame, or none, and rate-limits how often the
same one can re-fire.

Ported from the Watchdog wearable's vision/hazard.py, minus everything that
was about a wearable with a speaker: audio sequencing, the JSONL event log,
per-alert frame snapshots, mute state, and the per-class filter. What's left
is the arbitration itself -- which of the tracked people (if any) is the one
worth reacting to right now.

PERSON-ONLY, deliberately. Watchdog also ranked bicycle/car/bus/truck/dog;
webcam_tracker's detector filters to the person class, and Detection carries
no label at all, so there is nothing else here to rank. Adding other classes
is a detector change, not a change to this file.

The output is a Hazard, not an action. What to do about it -- alert, slow,
climb, hold position -- belongs to whatever consumes this, so that a drone
and a wearable can share the ranking without sharing a response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from webcam_tracker.config.settings import HazardConfig
from webcam_tracker.hazard.ttc import TrackStore, compute_ttc, zone_from_cx

if TYPE_CHECKING:
    # Typing only. TrackedPerson is a plain dataclass, but reaching it at
    # runtime -- by either path, package or submodule -- executes
    # tracking/__init__, which imports PersonTracker and therefore ultralytics
    # and torch. Nothing here needs them: this module reads four floats and an
    # int off whatever it's handed. Keeping the import out of the runtime path
    # is what lets the hazard logic load with numpy alone, on hardware with no
    # detector attached at all.
    from webcam_tracker.tracking import TrackedPerson


@dataclass(frozen=True)
class Hazard:
    """The single person judged most worth reacting to on one frame."""

    track_id: int
    zone: str
    ttc: float | None
    urgent: bool

    @property
    def is_static(self) -> bool:
        """True when this was picked by the close-but-not-closing fallback.

        ttc is None precisely when looming couldn't be measured, so the two
        are the same fact -- but callers reading `is_static` say what they
        mean, and a static hazard is never urgent (see below).
        """
        return self.ttc is None


class HazardMonitor:
    """Stateful, one instance per run: owns box history and alert cooldowns.

    Call update() once per frame with every tracked person. Feeding it a
    filtered subset breaks the cooldowns (a track that vanishes and returns
    looks new) and the TTC history (gaps splice unrelated positions).
    """

    def __init__(self, config: HazardConfig) -> None:
        self._config = config
        self._store = TrackStore(config.sample_window)
        # Two maps, because escalation must be able to break a cooldown that a
        # weaker report set -- see _is_in_cooldown.
        self._last_report: dict[int, float] = {}  # any hazard, per track
        self._last_moving_report: dict[int, float] = {}  # measured-TTC hazards only
        self._last_alert = float("-inf")

    def update(
        self,
        tracked_people: list[TrackedPerson],
        frame_width: int,
        frame_height: int,
        now: float | None = None,
    ) -> Hazard | None:
        """Returns the hazard to react to this frame, or None.

        None covers three different situations on purpose -- nothing is
        approaching, something is but isn't close enough yet, and something
        is but we just reported it -- because no caller needs to tell them
        apart to decide what to do next.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"frame size must be positive, got {frame_width}x{frame_height}")
        now = time.monotonic() if now is None else now
        cfg = self._config

        candidates: list[Hazard] = []
        active_ids: set[int] = set()

        for person in tracked_people:
            active_ids.add(person.track_id)

            # Normalize here and nowhere else. Detection/TrackedPerson are in
            # pixels; every threshold below is a fraction of the frame. Pixels
            # reaching the thresholds wouldn't raise -- min_width would pass
            # for every box and every cx would land past right_min -- so this
            # boundary is the whole defense against that.
            width = person.width / frame_width
            cx = person.center[0] / frame_width
            bottom_y = person.y2 / frame_height

            samples = self._store.update(person.track_id, now, width)
            zone = zone_from_cx(
                cx,
                discard_margin=cfg.zone_discard_margin,
                left_max=cfg.zone_left_max,
                right_min=cfg.zone_right_min,
            )
            if zone is None:
                continue

            ttc = compute_ttc(
                samples,
                min_samples=cfg.min_samples,
                min_width=cfg.min_width_fraction,
                ttc_min=cfg.ttc_min,
                ttc_max=cfg.ttc_max,
            )
            if ttc is not None:
                if ttc <= cfg.ttc_alert:
                    candidates.append(
                        Hazard(person.track_id, zone, ttc, urgent=ttc < cfg.ttc_urgent)
                    )
            elif bottom_y >= cfg.static_bottom_y:
                # Not measurably closing (stationary, or too few samples yet)
                # but already filling the bottom of frame, so close enough to
                # matter anyway. Never urgent: we don't know its closing rate,
                # only its distance, and guessing urgency from distance alone
                # is how a parked obstacle outranks a real approach.
                candidates.append(Hazard(person.track_id, zone, None, urgent=False))

        self._store.prune(active_ids)
        if not candidates:
            return None

        chosen = self._rank(candidates)
        if self._is_in_cooldown(chosen, now):
            return None
        if now - self._last_alert < cfg.global_gap:
            # Global rate limit, deliberately absolute: even an urgent hazard
            # waits out the remainder. Known ceiling -- it can delay an urgent
            # report by up to global_gap seconds. Tune global_gap for the
            # drone rather than adding an urgency bypass here, so there stays
            # exactly one place that decides how often this can speak.
            return None

        self._last_report[chosen.track_id] = now
        if not chosen.is_static:
            self._last_moving_report[chosen.track_id] = now
        self._last_alert = now
        return chosen

    def _is_in_cooldown(self, chosen: Hazard, now: float) -> bool:
        """Rate-limit repeats per track, but let a static report be escalated.

        Asymmetric on purpose. A track's FIRST report is very often the static
        one: a person entering low in frame has no measurable looming until
        min_samples frames have passed (~0.54s at the Pi's 9 FPS, vs ~0.21s on
        the NPU this was ported from), yet may already be filling the bottom of
        frame. Gating a later measured -- possibly urgent -- TTC on the cooldown
        that static report set would swallow it for track_cooldown seconds,
        which is most of the time available to react.

        So a measured hazard only waits on the last MEASURED report, while a
        static one waits on the last report of any kind. Escalation gets
        through; repetition and de-escalation don't.
        """
        if chosen.is_static:
            last = self._last_report.get(chosen.track_id, float("-inf"))
            return now - last < self._config.static_cooldown
        last = self._last_moving_report.get(chosen.track_id, float("-inf"))
        return now - last < self._config.track_cooldown

    @staticmethod
    def _rank(candidates: list[Hazard]) -> Hazard:
        """Soonest real collision wins; a static hazard only wins if it's alone.

        A measured approach always outranks a close-but-static object no
        matter how close that object is -- we know one is closing and only
        that the other is near. Ties among static candidates break on lowest
        track_id purely so the choice is deterministic frame to frame.
        """
        moving = [c for c in candidates if c.ttc is not None]
        if moving:
            return min(moving, key=lambda c: c.ttc)  # type: ignore[arg-type,return-value]
        return min(candidates, key=lambda c: c.track_id)
