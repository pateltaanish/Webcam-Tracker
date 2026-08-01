"""Live identity tracking preview (Stage 2.4): follow a registered PERSON.

The whole Stage 1 pipeline (detect -> track -> select -> gimbal -> recovery ->
state machine) driven by IDENTITY instead of a mouse click: you pick a
registered person, and the system follows that specific person -- and only
them -- reacquiring them by face if they leave and return (even as a new track
id), while refusing to lock onto anyone else.

Compared to preview_state_machine.py, selection is `select_person(id)` and the
state machine is given the frame image so identity can match faces. Each track
is labeled with its recognized identity; the state banner shows the same
IDLE/TRACKING/OCCLUDED/SEARCH states, now gated by identity.

Run from the repo root (after registering someone with register_person.py):
    .venv\\Scripts\\python.exe scripts\\preview_identity_tracking.py

Controls: 'e'/'r' emergency stop/resume, 'q'/Esc quit. Requires the identity
dependencies (requirements-identity.txt).
"""

from __future__ import annotations

import getpass

import cv2
import numpy as np

from webcam_tracker.config import load_config
from webcam_tracker.database import InvalidPassphraseError, Person, create_profile_store
from webcam_tracker.detection import create_detector
from webcam_tracker.face_recognition import create_face_embedder, create_face_matcher
from webcam_tracker.identity import IdentityTracker, create_identity_tracker
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.state_machine import create_state_machine
from webcam_tracker.tracking import TrackedPerson, create_tracker
from webcam_tracker.video_input import VideoSourceError, create_source
from webcam_tracker.visualization import (
    draw_gimbal_widget,
    draw_perf_overlay,
    draw_recovery_overlay,
    draw_state_banner,
    draw_target_overlay,
    draw_tracked_people,
)

WINDOW_NAME = "webcam_tracker identity tracking"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _choose_person(people: list[Person]) -> Person:
    if len(people) == 1:
        return people[0]
    print("\nRegistered people:")
    for i, person in enumerate(people):
        print(f"  [{i}] {person.display_name}")
    while True:
        raw = input("Track which number? ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(people):
            return people[int(raw)]
        print("Enter one of the numbers above.")


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)
    logger = get_logger(__name__)

    store = create_profile_store(config)
    if not store.is_initialized():
        raise SystemExit("No identity store yet -- register someone first with register_person.py")
    try:
        store.unlock(getpass.getpass("Operator passphrase: "))
    except InvalidPassphraseError:
        raise SystemExit("Wrong passphrase -- the store did not unlock.") from None

    try:
        people = store.list_people()
        if not people:
            raise SystemExit("No registered people to track.")
        target = _choose_person(people)

        print("Loading face model...")
        embedder = create_face_embedder(config)
        embedder.load()
        matcher = create_face_matcher(config, store)
        identity = create_identity_tracker(config, embedder, matcher)
        state_machine = create_state_machine(config, identity=identity)
        state_machine.select_person(target.id)
        print(f"Tracking '{target.display_name}'. Everyone else is ignored.")

        detector = create_detector(config)
        detector.load()
        tracker = create_tracker(config)
        perf = PerfMonitor(fps_window_seconds=config.perf_monitor.fps_window_seconds)

        source = create_source(config)
        cv2.namedWindow(WINDOW_NAME)
        try:
            with source:
                for frame in source:
                    perf.start_frame()
                    with perf.measure("detection"):
                        detections = detector.detect(frame.image)
                    with perf.measure("tracking"):
                        tracked = tracker.update(detections)
                    perf.end_frame()

                    height, width = frame.image.shape[:2]
                    status = state_machine.update(tracked, width, height, image=frame.image)

                    image = frame.image.copy()
                    draw_tracked_people(
                        image, tracked, identity.stable_ids(t.track_id for t in tracked)
                    )
                    _draw_identities(image, tracked, identity)
                    draw_target_overlay(image, status.target_status)
                    draw_recovery_overlay(
                        image,
                        status.recovery_status,
                        reacquire_radius_fraction=config.recovery.reacquire_radius_fraction,
                    )
                    draw_gimbal_widget(
                        image,
                        status.gimbal_command,
                        pan_limit_deg=config.gimbal.pan.angle_limit_deg,
                        tilt_limit_deg=config.gimbal.tilt.angle_limit_deg,
                    )
                    draw_state_banner(image, status)
                    draw_perf_overlay(
                        image, perf.snapshot(), extra_text=f"target: {target.display_name}"
                    )
                    cv2.imshow(WINDOW_NAME, image)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("e"):
                        state_machine.emergency_stop()
                    elif key == ord("r"):
                        state_machine.resume()
        except VideoSourceError as exc:
            logger.error("Could not open video source", extra={"error": str(exc)})
            raise SystemExit(1) from exc
        finally:
            cv2.destroyAllWindows()
    finally:
        store.close()


def _draw_identities(
    image: np.ndarray, tracked: list[TrackedPerson], identity: IdentityTracker
) -> None:
    for person in tracked:
        result = identity.identity_of(person.track_id)
        if result is None:
            continue
        if result.is_confirmed:
            text = f"{result.display_name} {result.confidence:.0%}"
            color = (0, 255, 0)
        else:
            text = "verifying..."
            color = (0, 255, 255)
        cv2.putText(image, text, (int(person.x1), int(person.y2) + 18), _FONT, 0.6, color, 2)


if __name__ == "__main__":
    main()
