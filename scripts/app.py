"""Unified webcam_tracker console with multi-user login (Stage 2.5).

Every launch begins with an AUTH phase -- you must Log in or Enroll before you
reach the tracker:

    ===== webcam_tracker =====
      [1] Log in
      [2] Enroll as new user
      [q] Quit

Two account modes, chosen once when the very first user is enrolled (the whole
store commits to one or the other):

  * SHARED KEY   -- one shared passphrase unlocks the store. Names are unique
                    login labels; after login you can target ANYONE enrolled
                    (the tracker's original behaviour). Joining needs the shared
                    key.
  * PER USER     -- each user has their OWN passphrase protecting their OWN face
                    data. After login the tracker follows only YOU. Anyone can
                    enroll a fresh account, and if you forget your passphrase you
                    can delete just your account and enroll again.

After login, a SHARED-key store additionally asks how *this session* should be
scoped -- shared (see everyone, pick a target) or personal (restrict the session
to your own data, tracker follows only you). That choice is per login, not
permanent, and is a convenience scope rather than a security boundary: the
shared DEK still decrypts everything. A per_user login is always personal --
its DEK cannot decrypt anyone else's rows, so there is nothing to choose.

Then you get the tracking menu (preview tracking, change passphrase, log out,
quit). Tracking/targeting/gimbal/recovery are unchanged from before.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\app.py

Requires the identity dependencies:
    .venv\\Scripts\\python.exe -m pip install -r requirements-identity.txt
"""

from __future__ import annotations

import getpass

import cv2
import numpy as np

from webcam_tracker.config import AppConfig, load_config
from webcam_tracker.database import (
    MODE_PER_USER,
    MODE_SHARED,
    AccountError,
    AccountManager,
    InvalidPassphraseError,
    Login,
    NameTakenError,
    NoSuchAccountError,
    Person,
    ProfileStore,
    WeakPassphraseError,
    create_account_manager,
    create_profile_store,
)
from webcam_tracker.detection import create_detector
from webcam_tracker.face_recognition import (
    DetectedFace,
    FaceEmbedder,
    create_face_embedder,
    create_face_matcher,
)
from webcam_tracker.identity import IdentityTracker, create_identity_tracker
from webcam_tracker.logging_utils import configure_logging, get_logger
from webcam_tracker.perf_monitor import PerfMonitor
from webcam_tracker.registration import SampleEvaluation, create_registrar
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

logger = get_logger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_REGISTER_WINDOW = "webcam_tracker enrollment"
_TRACKING_WINDOW = "webcam_tracker identity tracking"

_ANGLE_PROMPTS = [
    "look straight at the camera",
    "turn your head slightly LEFT",
    "turn your head slightly RIGHT",
    "tilt your chin slightly UP",
    "tilt your chin slightly DOWN",
]
_CONSENT_TEXT = (
    "\nCONSENT\n"
    "Enrolling stores your face *embeddings* (not images) for consent-based\n"
    "tracking, encrypted under your passphrase. You can delete them at any time.\n"
)


class EmbedderCache:
    """Loads the ~280 MB face model at most once per process and shares it
    across enrollments and tracking sessions."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._embedder: FaceEmbedder | None = None

    def get(self) -> FaceEmbedder:
        if self._embedder is None:
            print("Loading face model (first use downloads ~280 MB)...")
            embedder = create_face_embedder(self._config)
            embedder.load()
            self._embedder = embedder
        return self._embedder


# ===========================================================================
# AUTH PHASE
# ===========================================================================
def _authenticate(
    config: AppConfig, accounts: AccountManager, embedders: EmbedderCache
) -> Login | None:
    """Drive login/enroll until a user authenticates. Returns the Login, or
    None if the user quits."""
    while True:
        if not accounts.exists():
            print("\nNo users registered yet. Create the first account to set up the store.")
            print("  [1] Enroll as new user\n  [q] Quit")
            choice = input("Choose: ").strip().lower()
            if choice == "1":
                login = _enroll_flow(config, accounts, embedders, first=True)
                if login is not None:
                    return login
            elif choice in ("q", "quit", "exit"):
                return None
            else:
                print("Not a valid choice.")
            continue

        print("\n===== webcam_tracker =====")
        print("  [1] Log in")
        print("  [2] Enroll as new user")
        print("  [q] Quit")
        choice = input("Choose: ").strip().lower()
        if choice == "1":
            login = _login_flow(config, accounts)
            if login is not None:
                return login
        elif choice == "2":
            login = _enroll_flow(config, accounts, embedders, first=False)
            if login is not None:
                return login
        elif choice in ("q", "quit", "exit"):
            return None
        else:
            print("Not a valid choice.")


def _login_flow(config: AppConfig, accounts: AccountManager) -> Login | None:
    name = input("Username: ").strip()
    if not name:
        return None
    if not accounts.has_name(name):
        print(f"No account named '{name}'.")
        return None
    passphrase = getpass.getpass("Passphrase: ")
    try:
        login = accounts.login(name, passphrase)
    except InvalidPassphraseError:
        print("Wrong passphrase.")
        _offer_forgot(config, accounts, name)
        return None
    except NoSuchAccountError:
        print(f"No account named '{name}'.")
        return None
    print(f"\nWelcome back, {login.display_name}.")
    return login


def _offer_forgot(config: AppConfig, accounts: AccountManager, name: str) -> None:
    """Forgotten-passphrase recovery. Only meaningful in per_user mode, where an
    account owns its own key -- deleting it and re-enrolling is the only reset
    (the passphrase itself is unrecoverable by design). In shared mode the
    passphrase belongs to everyone, so there's nothing personal to reset."""
    if accounts.mode == MODE_SHARED:
        print(
            "This store uses a SHARED key -- the passphrase is the same for every\n"
            "user, so it can't be reset by deleting your name. If someone still\n"
            "knows it, ask them. Otherwise the store must be recreated from scratch."
        )
        return
    if input("Forgot it? Delete this account and re-enroll? (yes/no): ").strip().lower() != "yes":
        return
    print(
        f"\nThis permanently deletes '{name}'s account AND face data -- it cannot\n"
        "be recovered (the encryption key is gone)."
    )
    if input(f"Type the username '{name}' to confirm: ").strip() != name:
        print("Names didn't match -- nothing deleted.")
        return
    try:
        user_id = accounts.delete_account(name)
    except NoSuchAccountError:
        print("Account no longer exists.")
        return
    store = create_profile_store(config)
    store.purge_person(user_id)  # dek-less delete of the orphaned data
    store.close()
    print(f"'{name}' deleted. Enroll again as a new user when ready.")


def _enroll_flow(
    config: AppConfig, accounts: AccountManager, embedders: EmbedderCache, first: bool
) -> Login | None:
    """Create an account (choosing the store mode if this is the first user),
    then run the face scan. Rolls the account back if the scan is aborted so a
    credential never lingers without face data."""
    mode = _choose_mode() if first else accounts.mode
    if mode is None:
        return None

    name = _prompt_unique_name(accounts)
    if name is None:
        return None

    print(_CONSENT_TEXT)
    if input(f"Does {name} consent to enrollment? (yes/no): ").strip().lower() != "yes":
        print("Consent not given -- nothing stored.")
        return None

    passphrase = _prompt_new_passphrase(config, mode, first)
    if passphrase is None:
        return None

    try:
        login = (
            accounts.create(mode, name, passphrase) if first else accounts.enroll(name, passphrase)
        )
    except WeakPassphraseError as exc:
        print(f"Passphrase rejected: {exc}")
        return None
    except NameTakenError:
        print(f"'{name}' was just taken -- try again.")
        return None
    except InvalidPassphraseError:
        # Shared mode: the passphrase must BE the shared key to join.
        print("That isn't the shared key for this store -- ask an existing user for it.")
        return None
    except AccountError as exc:
        print(f"Could not create the account: {exc}")
        return None

    # Face scan, stored under this account's key and tied to its user_id.
    faces = _capture_faces(config, login, embedders)
    if faces is None:
        # Roll back: drop the just-created credential + any (empty) data. If this
        # was the FIRST account, remove the whole store so the mode isn't locked.
        if first:
            accounts.destroy()
        else:
            accounts.delete_account(name)
        store = create_profile_store(config)
        store.purge_person(login.user_id)
        store.close()
        print("Enrollment cancelled -- account not created.")
        return None

    store = create_profile_store(config)
    store.attach(login.dek, login.scope_person_id)
    try:
        registrar = create_registrar(config, store, embedders.get())
        person = registrar.enroll(
            login.display_name,
            faces,
            consent_note="self-enrolled, unified app",
            person_id=login.user_id,
        )
        print(f"\nEnrolled '{person.display_name}' with {len(faces)} face samples.")
    finally:
        store.close()
    return login


def _choose_mode() -> str | None:
    print(
        "\nThis is the FIRST account -- pick how the whole store will work "
        "(permanent for this store):\n"
        "  [1] Shared key  -- one shared passphrase; after login you can track "
        "ANYONE enrolled.\n"
        "  [2] Per user    -- each user has their own passphrase and own data; "
        "after login the tracker follows only YOU."
    )
    while True:
        choice = input("Mode [1/2] (blank to cancel): ").strip()
        if choice == "":
            return None
        if choice == "1":
            return MODE_SHARED
        if choice == "2":
            return MODE_PER_USER
        print("Enter 1, 2, or blank.")


def _prompt_unique_name(accounts: AccountManager) -> str | None:
    while True:
        name = input("Choose a username (blank to cancel): ").strip()
        if name == "":
            return None
        if accounts.has_name(name):
            print(f"'{name}' is already taken -- names must be unique.")
            continue
        return name


def _prompt_new_passphrase(config: AppConfig, mode: str, first: bool) -> str | None:
    minimum = config.identity.min_passphrase_length
    if mode == MODE_SHARED and not first:
        # Joining a shared store: enter the EXISTING shared key (not a new one).
        return getpass.getpass("Enter the store's shared passphrase: ") or None
    prompt = (
        f"Set the SHARED passphrase (min {minimum} chars): "
        if mode == MODE_SHARED
        else f"Set your passphrase (min {minimum} chars): "
    )
    first_entry = getpass.getpass(prompt)
    if first_entry == "":
        return None
    if getpass.getpass("Confirm passphrase: ") != first_entry:
        print("Passphrases didn't match.")
        return None
    return first_entry


def _capture_faces(
    config: AppConfig, login: Login, embedders: EmbedderCache
) -> list[DetectedFace] | None:
    """Interactive webcam capture of quality-checked face samples. Returns the
    captured samples, or None if aborted. Uses a throwaway store handle just to
    build the registrar/quality evaluator (no writes here)."""
    embedder = embedders.get()
    scratch = create_profile_store(config)
    scratch.attach(login.dek, login.scope_person_id)
    try:
        registrar = create_registrar(config, scratch, embedder)
        needed = config.registration.samples_required
        captured: list[DetectedFace] = []
        try:
            source = create_source(config)
        except VideoSourceError as exc:
            print(f"Could not open camera: {exc}")
            return None
        cv2.namedWindow(_REGISTER_WINDOW)
        try:
            with source:
                for frame in source:
                    evaluation = registrar.evaluate(frame.image)
                    prompt = _ANGLE_PROMPTS[len(captured) % len(_ANGLE_PROMPTS)]
                    image = frame.image.copy()
                    _draw_capture_feedback(image, evaluation, len(captured), needed, prompt)
                    cv2.imshow(_REGISTER_WINDOW, image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        return None
                    if key == ord(" ") and evaluation.accepted and evaluation.face is not None:
                        captured.append(evaluation.face)
                        print(f"captured sample {len(captured)}/{needed}")
                        if len(captured) >= needed:
                            return captured
        except VideoSourceError as exc:
            print(f"Camera error: {exc}")
            return None
        finally:
            cv2.destroyWindow(_REGISTER_WINDOW)
            cv2.waitKey(1)
    finally:
        scratch.close()
    return None


def _draw_capture_feedback(
    image: np.ndarray, evaluation: SampleEvaluation, captured: int, needed: int, prompt: str
) -> None:
    if evaluation.face is not None:
        face = evaluation.face
        color = (0, 255, 0) if evaluation.accepted else (0, 165, 255)
        cv2.rectangle(image, (int(face.x1), int(face.y1)), (int(face.x2), int(face.y2)), color, 2)
    status = "GOOD -- press SPACE" if evaluation.accepted else " / ".join(evaluation.reasons)
    banner_color = (0, 255, 0) if evaluation.accepted else (0, 165, 255)
    cv2.putText(image, f"[{captured}/{needed}] {prompt}", (10, 30), _FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(image, status, (10, 60), _FONT, 0.7, banner_color, 2)
    cv2.putText(
        image,
        "SPACE=capture  q/Esc=abort",
        (10, image.shape[0] - 15),
        _FONT,
        0.5,
        (200, 200, 200),
        1,
    )


# ===========================================================================
# POST-LOGIN SESSION
# ===========================================================================
def _session_menu(
    config: AppConfig,
    accounts: AccountManager,
    login: Login,
    embedders: EmbedderCache,
) -> bool:
    """Menu shown after login. Returns True if the user chose to quit the whole
    app, False to log out (back to the auth phase)."""
    personal = _choose_session_scope(login)
    if personal is None:
        return False  # cancelled at the scope prompt -- back to the auth phase
    store = create_profile_store(config)
    # A personal session pins the store to this one person even on a shared-key
    # store, so listing, matching and targeting can't see anyone else. Every
    # restriction downstream follows from this one scope.
    store.attach(login.dek, login.user_id if personal else None)
    try:
        while True:
            label = "personal" if personal else "shared"
            print(f"\n----- {login.display_name} ({label} session) -----")
            print("  [1] Preview tracking")
            print("  [2] Change my passphrase")
            if not personal:
                print("  [3] List registered people")
            if login.mode == MODE_SHARED:
                print("  [p] Set a personal passphrase (requires the shared passphrase)")
            print("  [l] Log out")
            print("  [q] Quit")
            choice = input("Choose: ").strip().lower()
            if choice == "1":
                _tracking_flow(config, store, login, embedders, personal)
            elif choice == "2":
                _change_passphrase_flow(config, accounts, login)
            elif choice == "3" and not personal:
                _list_flow(store)
            elif choice == "p" and login.mode == MODE_SHARED:
                _set_personal_passphrase_flow(config, accounts, login)
            elif choice == "l":
                return False
            elif choice in ("q", "quit", "exit"):
                return True
            else:
                print("Not a valid choice.")
    finally:
        store.close()


def _choose_session_scope(login: Login) -> bool | None:
    """Ask, once per login, whether this session sees everyone or only the
    logged-in user. Returns True for personal, False for shared, None to cancel
    (which drops back to the auth phase).

    Only a shared-key store gets the choice. A per_user login's DEK can only
    decrypt that user's own rows, so "shared" isn't merely disallowed there --
    there would be nothing to decrypt -- and it is forced personal.

    Note this is a convenience scope, NOT a security boundary: on a shared store
    the DEK still decrypts everything, so a personal session is a self-imposed
    restriction that the same login can drop by choosing shared next time."""
    if login.mode != MODE_SHARED:
        print("\nThis store is PER USER -- this session can only ever see your own data.")
        return True
    print(
        f"\nHow should this session work, {login.display_name}?\n"
        "  [1] Shared    -- see everyone enrolled and pick who to track.\n"
        "  [2] Personal  -- restrict this session to your own data; the tracker "
        "follows only YOU."
    )
    while True:
        choice = input("Session [1/2] (blank to log out): ").strip()
        if choice == "":
            return None
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("Enter 1, 2, or blank.")


def _change_passphrase_flow(config: AppConfig, accounts: AccountManager, login: Login) -> None:
    has_personal = login.mode == MODE_SHARED and accounts.has_personal_passphrase(
        login.display_name
    )
    if login.mode == MODE_SHARED and not has_personal:
        print(
            "\nHeads up: this store uses a SHARED key. Changing it changes the\n"
            "passphrase for EVERY user -- they'll all need the new one."
        )
    current = getpass.getpass(
        "Current personal passphrase: " if has_personal else "Current passphrase: "
    )
    minimum = config.identity.min_passphrase_length
    new = getpass.getpass(f"New passphrase (min {minimum} chars): ")
    if getpass.getpass("Confirm new passphrase: ") != new:
        print("Passphrases didn't match -- nothing changed.")
        return
    try:
        accounts.change_passphrase(login.display_name, current, new)
    except InvalidPassphraseError:
        print("Current passphrase was wrong -- nothing changed.")
        return
    except WeakPassphraseError as exc:
        print(f"New passphrase rejected: {exc} -- nothing changed.")
        return
    print("Passphrase changed.")


def _set_personal_passphrase_flow(
    config: AppConfig, accounts: AccountManager, login: Login
) -> None:
    """Shared-key stores only: give this login its own passphrase that
    afterwards overrides the shared key for logging in as this name -- so a
    device with one shared secret can still give each user a private one.
    Requires the store's CURRENT shared passphrase every time (proof you
    already hold the one secret everyone here shares)."""
    if accounts.has_personal_passphrase(login.display_name):
        print("\nYou already have a personal passphrase set -- this replaces it.")
    else:
        print(
            "\nThis sets YOUR OWN passphrase for logging in as "
            f"'{login.display_name}'. Afterwards it replaces the shared\n"
            "passphrase for your name -- the shared passphrase alone will no "
            "longer log you in."
        )
    shared_passphrase = getpass.getpass("Enter the store's shared passphrase: ")
    minimum = config.identity.min_passphrase_length
    new = getpass.getpass(f"Set your personal passphrase (min {minimum} chars): ")
    if getpass.getpass("Confirm personal passphrase: ") != new:
        print("Passphrases didn't match -- nothing changed.")
        return
    try:
        accounts.set_personal_passphrase(login.display_name, shared_passphrase, new)
    except InvalidPassphraseError:
        print("That wasn't the store's shared passphrase -- nothing changed.")
        return
    except WeakPassphraseError as exc:
        print(f"Passphrase rejected: {exc} -- nothing changed.")
        return
    print("Personal passphrase set. Log in with it instead of the shared passphrase from now on.")


def _list_flow(store: ProfileStore) -> None:  # shared mode only
    people = store.list_people()
    print(f"\n{len(people)} registered:")
    for person in people:
        print(f"  - {person.display_name}  (since {person.created_at})")


def _tracking_flow(
    config: AppConfig,
    store: ProfileStore,
    login: Login,
    embedders: EmbedderCache,
    personal: bool,
) -> None:
    people = store.list_people()
    if not people:
        print("No enrolled face data to track.")
        return
    # Personal session: the store is scoped, so the only person listed is you.
    target: Person | None = people[0] if personal else _choose_person(people)
    if target is None:
        return  # backed out of the person picker

    embedder = embedders.get()
    matcher = create_face_matcher(config, store)  # loads templates the DEK can see
    identity = create_identity_tracker(config, embedder, matcher)
    state_machine = create_state_machine(config, identity=identity)
    state_machine.select_person(target.id)
    print(f"Tracking '{target.display_name}'. Everyone else is ignored.")

    detector = create_detector(config)
    detector.load()
    tracker = create_tracker(config)
    perf = PerfMonitor(fps_window_seconds=config.perf_monitor.fps_window_seconds)

    try:
        source = create_source(config)
    except VideoSourceError as exc:
        print(f"Could not open camera: {exc}")
        return
    cv2.namedWindow(_TRACKING_WINDOW)
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
                stable_ids = identity.stable_ids(t.track_id for t in tracked)
                draw_tracked_people(image, tracked, stable_ids)
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
                cv2.imshow(_TRACKING_WINDOW, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e"):
                    state_machine.emergency_stop()
                elif key == ord("r"):
                    state_machine.resume()
    except VideoSourceError as exc:
        print(f"Camera error: {exc}")
    finally:
        cv2.destroyWindow(_TRACKING_WINDOW)
        cv2.waitKey(1)


def _choose_person(people: list[Person]) -> Person | None:
    if len(people) == 1:
        return people[0]
    print("\nRegistered people:")
    for i, person in enumerate(people):
        print(f"  [{i}] {person.display_name}")
    while True:
        raw = input("Track which number? (blank to cancel) ").strip()
        if raw == "":
            return None
        if raw.isdigit() and 0 <= int(raw) < len(people):
            return people[int(raw)]
        print("Enter one of the numbers above, or blank to cancel.")


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


# ===========================================================================
def main() -> None:
    config = load_config()
    configure_logging(level=config.logging.level, json_format=False)
    accounts = create_account_manager(config)
    embedders = EmbedderCache(config)

    while True:
        login = _authenticate(config, accounts, embedders)
        if login is None:
            break
        quit_app = _session_menu(config, accounts, login, embedders)
        if quit_app:
            break
    print("Bye.")


if __name__ == "__main__":
    main()
