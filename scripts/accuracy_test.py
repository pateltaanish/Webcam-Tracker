"""Accuracy test harness (Stage 2.6): measure FAR/FRR on your own captured data.

Feeds a folder of test photos through the same face embedder + matcher used at
runtime and reports False Accept Rate / False Reject Rate -- the numbers
docs/01_risks_and_assumptions.md R5 calls for before trusting a match
threshold as "final," not a placeholder. Also sweeps a range of thresholds so
you can see where FAR and FRR cross, instead of guessing at one value.

Test data layout -- one folder per person, folder name = their exact
`display_name` as enrolled via `register_person.py`, plus one `unregistered/`
folder of photos of people who were NEVER enrolled:

    data/accuracy_test/
      Taanish/           <- known-target probes: photos of Taanish (genuine)
      Alex/              <- another enrolled person -- also covers
                            "wrong-registered-user": if a Taanish photo ever
                            matches Alex (or vice versa), that shows up below
                            as a failed genuine attempt, same as UNKNOWN would
      unregistered/       <- stranger photos; nobody in this folder is enrolled
                            (impostor attempts -- covers "unregistered-person")

Run from the repo root, after registering at least the people you're testing:
    .venv\\Scripts\\python.exe scripts\\accuracy_test.py [data_dir]

(data_dir defaults to data/accuracy_test.) Requires the identity dependencies
(requirements-identity.txt).
"""

from __future__ import annotations

import getpass
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2

from webcam_tracker.config import load_config
from webcam_tracker.database import InvalidPassphraseError, create_profile_store
from webcam_tracker.face_recognition import (
    FaceEmbedder,
    FaceMatcher,
    create_face_embedder,
    create_face_matcher,
)
from webcam_tracker.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
_UNREGISTERED_LABEL = "unregistered"
_THRESHOLD_SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(13)]  # 0.20 .. 0.80


@dataclass(frozen=True)
class Probe:
    path: Path
    expected_label: str  # an enrolled display_name, or "unregistered"
    is_genuine: bool  # expected_label names someone actually enrolled
    predicted_label: str | None  # best-matching template's display_name
    score: float  # best raw cosine similarity, independent of any threshold


def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/accuracy_test")
    if not data_dir.is_dir():
        raise SystemExit(
            f"No test data at {data_dir} -- create it first (see this script's docstring "
            "for the expected folder layout)."
        )

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
            raise SystemExit("No registered people -- nothing to test against.")
        probes = list(_collect_probes(data_dir, embedder, matcher, matcher.enrolled_names))
        if not probes:
            raise SystemExit(f"No usable (readable, one-face) images found under {data_dir}.")

        genuine_count = sum(p.is_genuine for p in probes)
        impostor_count = len(probes) - genuine_count
        print(
            f"\n{len(probes)} probe(s): {genuine_count} genuine (known-target / "
            f"wrong-registered-user), {impostor_count} impostor (unregistered-person), "
            f"against {matcher.num_templates} template(s).\n"
        )

        print(f"At the configured threshold ({config.face.match_threshold:.2f}):")
        _print_report(probes, config.face.match_threshold)

        print("\nThreshold sweep (find where FAR and FRR cross to tune match_threshold):")
        print(f"{'threshold':>10} {'FAR':>8} {'FRR':>8} {'impostors':>10} {'genuine':>8}")
        for threshold in _THRESHOLD_SWEEP:
            far, frr, impostors, genuine = _far_frr(probes, threshold)
            print(f"{threshold:>10.2f} {far:>8.1%} {frr:>8.1%} {impostors:>10} {genuine:>8}")
    finally:
        store.close()


def _collect_probes(
    data_dir: Path, embedder: FaceEmbedder, matcher: FaceMatcher, enrolled_names: set[str]
) -> Iterator[Probe]:
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        is_genuine = label != _UNREGISTERED_LABEL
        if is_genuine and label not in enrolled_names:
            logger.warning(
                "Folder name doesn't match any enrolled person, skipping",
                extra={"folder": label},
            )
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Could not read image, skipping", extra={"path": str(image_path)})
                continue
            faces = embedder.detect(image)
            if not faces:
                logger.warning("No face detected, skipping", extra={"path": str(image_path)})
                continue
            # Largest face, in case a probe photo has a bystander in frame too.
            face = max(faces, key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1))
            best = matcher.best_candidate(face.embedding)
            yield Probe(
                path=image_path,
                expected_label=label,
                is_genuine=is_genuine,
                predicted_label=best.display_name,
                score=best.score,
            )


def _far_frr(probes: list[Probe], threshold: float) -> tuple[float, float, int, int]:
    """FAR/FRR at a given threshold, applied to each probe's raw score.

    Genuine probe (known-target/wrong-registered-user): correctly accepted
    only if the score clears `threshold` AND the accepted identity is the
    expected one -- a confident match to the WRONG enrolled person still
    counts as a failure here, same as UNKNOWN would.
    Impostor probe (unregistered-person): correctly rejected only if the
    score does NOT clear `threshold`, regardless of which template it's
    nearest to.
    """
    genuine = [p for p in probes if p.is_genuine]
    impostors = [p for p in probes if not p.is_genuine]
    false_rejects = sum(
        1 for p in genuine if not (p.score >= threshold and p.predicted_label == p.expected_label)
    )
    false_accepts = sum(1 for p in impostors if p.score >= threshold)
    far = false_accepts / len(impostors) if impostors else 0.0
    frr = false_rejects / len(genuine) if genuine else 0.0
    return far, frr, len(impostors), len(genuine)


def _print_report(probes: list[Probe], threshold: float) -> None:
    far, frr, impostor_count, genuine_count = _far_frr(probes, threshold)
    print(f"  FAR (impostors wrongly accepted): {far:.1%} ({impostor_count} probe(s))")
    print(f"  FRR (genuine probes wrongly rejected/misID'd): {frr:.1%} ({genuine_count} probe(s))")
    for p in probes:
        accepted = p.score >= threshold
        correct = (
            accepted and p.predicted_label == p.expected_label if p.is_genuine else not accepted
        )
        if not correct:
            outcome = f"matched '{p.predicted_label}'" if accepted else "UNKNOWN"
            print(f"    FAIL  {p.path}  want={p.expected_label} got={outcome} score={p.score:.3f}")


if __name__ == "__main__":
    main()
