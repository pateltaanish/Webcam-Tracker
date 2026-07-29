"""Unit tests for the multi-user login layer (Stage 2.5).

Covers both store modes end-to-end against the real crypto (with cheap Argon2
params for speed) and their integration with ProfileStore's DEK/scoping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from webcam_tracker.database.accounts import (
    MODE_PER_USER,
    MODE_SHARED,
    AccountError,
    AccountManager,
    NameTakenError,
    NoSuchAccountError,
)
from webcam_tracker.database.crypto import InvalidPassphraseError, KdfParams
from webcam_tracker.database.store import ProfileStore

CHEAP = KdfParams(time_cost=1, memory_kib=8192, parallelism=1)
PASS = "correcthorsebattery"  # >= 12 chars
PASS_B = "anotherlongpassphrase"


@pytest.fixture()
def accounts(tmp_path: Path) -> AccountManager:
    return AccountManager(tmp_path / "accounts.json", CHEAP, min_passphrase_length=12)


def _store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(tmp_path / "profiles.db", tmp_path / "unused.json", CHEAP, 12)


def _add_face(store: ProfileStore, name: str, person_id: str) -> None:
    person = store.add_person(name, consent_version="v1", person_id=person_id)
    assert person.id == person_id
    vec = np.ones(512, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    store.add_embedding(person.id, kind="face", model_id="test", vector=vec, quality=0.9)


# ------------------------------------------------------------------ shared mode
def test_shared_mode_join_requires_shared_key(accounts: AccountManager) -> None:
    accounts.create(MODE_SHARED, "Alice", PASS)
    with pytest.raises(InvalidPassphraseError):
        accounts.enroll("Bob", "not-the-shared-key")
    bob = accounts.enroll("Bob", PASS)  # correct shared key joins
    assert bob.scope_person_id is None
    assert accounts.login("Bob", PASS).dek == accounts.login("Alice", PASS).dek


def test_shared_mode_sees_everyone(accounts: AccountManager, tmp_path: Path) -> None:
    alice = accounts.create(MODE_SHARED, "Alice", PASS)
    store = _store(tmp_path)
    store.attach(alice.dek, alice.scope_person_id)
    _add_face(store, "Alice", alice.user_id)
    store.close()

    bob = accounts.enroll("Bob", PASS)
    store = _store(tmp_path)
    store.attach(bob.dek, bob.scope_person_id)
    _add_face(store, "Bob", bob.user_id)
    assert {p.display_name for p in store.list_people()} == {"Alice", "Bob"}
    store.close()


def test_shared_change_passphrase_rotates_for_all(accounts: AccountManager) -> None:
    accounts.create(MODE_SHARED, "Alice", PASS)
    accounts.enroll("Bob", PASS)
    accounts.change_passphrase("Alice", PASS, PASS_B)
    assert accounts.login("Bob", PASS_B) is not None
    with pytest.raises(InvalidPassphraseError):
        accounts.login("Bob", PASS)


# ----------------------------------------------------------------- per_user mode
def test_per_user_isolated_data(accounts: AccountManager, tmp_path: Path) -> None:
    alice = accounts.create(MODE_PER_USER, "Alice", PASS)
    assert alice.scope_person_id == alice.user_id
    store = _store(tmp_path)
    store.attach(alice.dek, alice.scope_person_id)
    _add_face(store, "Alice", alice.user_id)
    assert [p.display_name for p in store.list_people()] == ["Alice"]
    store.close()

    bob = accounts.enroll("Bob", PASS_B)  # own passphrase, no shared secret
    assert bob.dek != alice.dek
    store = _store(tmp_path)
    store.attach(bob.dek, bob.scope_person_id)
    _add_face(store, "Bob", bob.user_id)
    # Bob's scoped view is only Bob, even though Alice's row exists.
    assert [p.display_name for p in store.list_people()] == ["Bob"]
    store.close()


def test_per_user_wrong_passphrase_rejected(accounts: AccountManager) -> None:
    accounts.create(MODE_PER_USER, "Alice", PASS)
    accounts.enroll("Bob", PASS_B)
    with pytest.raises(InvalidPassphraseError):
        accounts.login("Alice", PASS_B)


def test_forgot_delete_and_reenroll(accounts: AccountManager, tmp_path: Path) -> None:
    accounts.create(MODE_PER_USER, "Alice", PASS)
    bob = accounts.enroll("Bob", PASS_B)
    store = _store(tmp_path)
    store.attach(bob.dek, bob.scope_person_id)
    _add_face(store, "Bob", bob.user_id)
    store.close()

    assert accounts.has_name("Bob")
    purged = accounts.delete_account("Bob")
    assert purged == bob.user_id
    assert not accounts.has_name("Bob")

    store = _store(tmp_path)
    store.purge_person(purged)  # dek-less delete
    store.close()

    bob2 = accounts.enroll("Bob", PASS)  # name free again, fresh account
    assert bob2.user_id != bob.user_id


# --------------------------------------------------------------------- general
def test_unique_names_case_insensitive(accounts: AccountManager) -> None:
    accounts.create(MODE_PER_USER, "Alice", PASS)
    assert accounts.has_name("alice")  # case-insensitive
    with pytest.raises(NameTakenError):
        accounts.enroll("ALICE", PASS_B)


def test_login_unknown_name(accounts: AccountManager) -> None:
    accounts.create(MODE_PER_USER, "Alice", PASS)
    with pytest.raises(NoSuchAccountError):
        accounts.login("Nobody", PASS)


def test_create_twice_fails(accounts: AccountManager) -> None:
    accounts.create(MODE_PER_USER, "Alice", PASS)
    with pytest.raises(AccountError):
        accounts.create(MODE_PER_USER, "Bob", PASS)


def test_mode_persisted(accounts: AccountManager) -> None:
    accounts.create(MODE_SHARED, "Alice", PASS)
    assert accounts.mode == MODE_SHARED
