"""Live face recognition preview (Stage 2.3): label faces as registered people.

Unlocks the encrypted identity store, then runs the webcam through the face
embedder + matcher: each detected face is labeled with the matched registered
person's name + similarity score (green), or UNKNOWN (red) when nothing clears
the match threshold. This is the payoff of registration -- it recognizes the
specific people you enrolled with scripts/register_person.py, and only them.

Run from the repo root (after registering at least one person):
    .venv\\Scripts\\python.exe scripts\\preview_face_match.py

Press 'q' or Esc to quit. Requires the identity dependencies
(requirements-identity.txt).
"""

from __future__ import annotations

import getpass

import cv2
import numpy as np

from webcam_tracker.config import load_config
from webcam_tracker.database import InvalidPassphraseError, create_profile_store
from webcam_tracker.face_recognition import (
    DetectedFace,
    MatchResult,
    create_face_embedder,
    create_face_matcher,
)
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.video_input import VideoSourceError, create_source

WINDOW_NAME = "webcam_tracker face recognition"
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    store = create_profile_store(config)
    if not store.is_initialized():
        raise SystemExit("No identity store yet -- register someone first with register_person.py")

    passphrase = getpass.getpass("Operator passphrase: ")
    try:
        store.unlock(passphrase)
    except InvalidPassphraseError:
        raise SystemExit("Wrong passphrase -- the store did not unlock.") from None

    try:
        print("Loading face model...")
        embedder = create_face_embedder(config)
        embedder.load()
        matcher = create_face_matcher(config, store)
        if matcher.num_templates == 0:
            print("WARNING: no registered people -- every face will show as UNKNOWN.")
        else:
            print(f"Loaded {matcher.num_templates} face template(s). Everyone else is UNKNOWN.")

        source = create_source(config)
        cv2.namedWindow(WINDOW_NAME)
        try:
            with source:
                for frame in source:
                    image = frame.image.copy()
                    for face in embedder.detect(frame.image):
                        _draw_match(image, face, matcher.match(face.embedding))
                    cv2.imshow(WINDOW_NAME, image)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
        except VideoSourceError as exc:
            raise SystemExit(f"Could not open camera: {exc}") from exc
        finally:
            cv2.destroyAllWindows()
    finally:
        store.close()


def _draw_match(image: np.ndarray, face: DetectedFace, result: MatchResult) -> None:
    color = _GREEN if result.is_match else _RED
    p1 = (int(face.x1), int(face.y1))
    p2 = (int(face.x2), int(face.y2))
    cv2.rectangle(image, p1, p2, color, 2)
    if result.is_match:
        label = f"{result.display_name} ({result.score:.2f})"
    else:
        label = f"UNKNOWN (best {result.score:.2f})"
    cv2.putText(image, label, (p1[0], max(0, p1[1] - 8)), _FONT, 0.6, color, 2)


if __name__ == "__main__":
    main()
