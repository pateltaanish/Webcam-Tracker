"""Interactive registration CLI (Stage 2.2): enroll a consenting person.

Consent-first, in-person only: you (the operator) run this with the person
physically present and consenting. It captures several quality-checked face
samples from the webcam, embeds them (ArcFace), and stores them in the
encrypted identity database with a consent record. No raw face images are
stored -- only the embeddings.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\register_person.py

You'll be asked for the operator passphrase (this creates the encrypted store
on first run, or unlocks it after), the person's display name, and an explicit
consent confirmation. Then a webcam window opens: line the person's face up so
the box turns green ("GOOD"), press SPACE to capture each sample (you'll be
prompted to vary the angle), or 'q'/Esc to abort.

Requires the identity dependencies:
    .venv\\Scripts\\python.exe -m pip install -r requirements-identity.txt
"""

from __future__ import annotations

import getpass

import cv2
import numpy as np

from webcam_tracker.config import AppConfig, load_config
from webcam_tracker.database import (
    InvalidPassphraseError,
    ProfileStore,
    WeakPassphraseError,
    create_profile_store,
)
from webcam_tracker.face_recognition import DetectedFace, create_face_embedder
from webcam_tracker.logging_utils import configure_logging
from webcam_tracker.registration import SampleEvaluation, create_registrar
from webcam_tracker.video_input import VideoSourceError, create_source

WINDOW_NAME = "webcam_tracker registration"
_ANGLE_PROMPTS = [
    "look straight at the camera",
    "turn your head slightly LEFT",
    "turn your head slightly RIGHT",
    "tilt your chin slightly UP",
    "tilt your chin slightly DOWN",
]
_CONSENT_TEXT = (
    "\nCONSENT\n"
    "This enrolls the person's face embeddings for consent-based tracking.\n"
    "The person must be present and agree. Face *images* are NOT stored -- only\n"
    "mathematical embeddings, encrypted. They can be deleted at any time.\n"
)


def _open_store(passphrase: str, config: AppConfig) -> ProfileStore:
    store = create_profile_store(config)
    if store.is_initialized():
        store.unlock(passphrase)
    else:
        print("No identity store yet -- creating a new encrypted store with this passphrase.")
        store.initialize(passphrase)
    return store


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    # 1) Operator passphrase -> unlock/create the encrypted store.
    passphrase = getpass.getpass("Operator passphrase: ")
    try:
        store = _open_store(passphrase, config)
    except InvalidPassphraseError:
        raise SystemExit("Wrong passphrase -- the store did not unlock.") from None
    except WeakPassphraseError as exc:
        raise SystemExit(f"Passphrase rejected: {exc}") from None

    try:
        # 2) Who, and explicit consent.
        display_name = input("Person's display name (a label/pseudonym is fine): ").strip()
        if not display_name:
            raise SystemExit("No name given -- aborting.")
        print(_CONSENT_TEXT)
        consent = input(f"Does {display_name} consent to enrollment? (yes/no): ").strip().lower()
        if consent != "yes":
            raise SystemExit("Consent not given -- aborting, nothing stored.")

        # 3) Load the face model (slow: downloads on first ever use).
        print("Loading face model (first run downloads ~280 MB)...")
        embedder = create_face_embedder(config)
        embedder.load()
        registrar = create_registrar(config, store, embedder)
        needed = config.registration.samples_required

        # 4) Capture loop.
        captured: list[DetectedFace] = []
        source = create_source(config)
        cv2.namedWindow(WINDOW_NAME)
        try:
            with source:
                for frame in source:
                    evaluation = registrar.evaluate(frame.image)
                    prompt = _ANGLE_PROMPTS[len(captured) % len(_ANGLE_PROMPTS)]
                    image = frame.image.copy()
                    _draw_feedback(image, evaluation, len(captured), needed, prompt)
                    cv2.imshow(WINDOW_NAME, image)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        raise SystemExit("Aborted -- nothing stored.")
                    if key == ord(" ") and evaluation.accepted and evaluation.face is not None:
                        captured.append(evaluation.face)
                        print(f"captured sample {len(captured)}/{needed}")
                        if len(captured) >= needed:
                            break
        except VideoSourceError as exc:
            raise SystemExit(f"Could not open camera: {exc}") from exc
        finally:
            cv2.destroyAllWindows()

        # 5) Enroll.
        person = registrar.enroll(display_name, captured, consent_note="in-person, operator CLI")
        print(f"\nEnrolled '{person.display_name}' (id={person.id}) with {len(captured)} samples.")
        print(f"Total registered people: {len(store.list_people())}")
    finally:
        store.close()


def _draw_feedback(
    image: np.ndarray, evaluation: SampleEvaluation, captured: int, needed: int, prompt: str
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if evaluation.face is not None:
        face = evaluation.face
        color = (0, 255, 0) if evaluation.accepted else (0, 165, 255)
        cv2.rectangle(image, (int(face.x1), int(face.y1)), (int(face.x2), int(face.y2)), color, 2)
    status = "GOOD -- press SPACE" if evaluation.accepted else " / ".join(evaluation.reasons)
    banner_color = (0, 255, 0) if evaluation.accepted else (0, 165, 255)
    cv2.putText(image, f"[{captured}/{needed}] {prompt}", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(image, status, (10, 60), font, 0.7, banner_color, 2)
    cv2.putText(
        image,
        "SPACE=capture  q/Esc=abort",
        (10, image.shape[0] - 15),
        font,
        0.5,
        (200, 200, 200),
        1,
    )


if __name__ == "__main__":
    main()
