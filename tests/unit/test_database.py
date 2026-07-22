"""Unit tests for webcam_tracker.database (Stage 2.1).

Covers the security-critical guarantees explicitly: a wrong passphrase fails
closed (no partial unlock), data is encrypted at rest (plaintext never appears
in the file), and delete really removes the biometric data. Uses deliberately
cheap Argon2 parameters so the KDF doesn't slow the test suite -- production
cost comes from config.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from webcam_tracker.database import (
    InvalidPassphraseError,
    KdfParams,
    PersonNotFoundError,
    ProfileStore,
    StoreAlreadyInitializedError,
    StoreNotInitializedError,
    WeakPassphraseError,
)
from webcam_tracker.database.crypto import (
    create_keyvault,
    decrypt_field,
    encrypt_field,
    unlock_keyvault,
)
from webcam_tracker.database.errors import DatabaseLockedError

# Cheap KDF for tests (min Argon2 memory is 8 KiB per lane). Production cost
# lives in configs/default.yaml, not here.
_FAST_KDF = KdfParams(time_cost=1, memory_kib=8192, parallelism=1)
_PASS = "correct-horse-battery-staple"
_OTHER = "another-good-passphrase"


def _store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(
        db_path=tmp_path / "profiles.db",
        keyvault_path=tmp_path / "keyvault.json",
        kdf_params=_FAST_KDF,
        min_passphrase_length=8,
    )


class TestCryptoPrimitives:
    def test_field_encrypt_decrypt_roundtrip(self) -> None:
        _, dek = create_keyvault(_PASS, _FAST_KDF)
        blob = encrypt_field(dek, b"secret bytes")
        assert blob != b"secret bytes"
        assert decrypt_field(dek, blob) == b"secret bytes"

    def test_decrypt_with_wrong_key_fails(self) -> None:
        _, dek = create_keyvault(_PASS, _FAST_KDF)
        blob = encrypt_field(dek, b"secret")
        with pytest.raises(InvalidPassphraseError):
            decrypt_field(os.urandom(32), blob)

    def test_keyvault_unlock_roundtrip(self) -> None:
        keyvault, dek = create_keyvault(_PASS, _FAST_KDF)
        assert unlock_keyvault(keyvault, _PASS) == dek

    def test_keyvault_wrong_passphrase_raises(self) -> None:
        keyvault, _ = create_keyvault(_PASS, _FAST_KDF)
        with pytest.raises(InvalidPassphraseError):
            unlock_keyvault(keyvault, _OTHER)

    def test_keyvault_has_no_secret_material(self) -> None:
        keyvault, _ = create_keyvault(_PASS, _FAST_KDF)
        # Salt/params/wrapped key are present; the passphrase and raw DEK are not.
        assert set(keyvault) >= {"salt", "dek_nonce", "wrapped_dek", "kdf_params"}
        assert _PASS not in str(keyvault)


class TestStoreLifecycle:
    def test_initialize_creates_files(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert not store.is_initialized()
        store.initialize(_PASS)
        assert store.is_initialized()
        assert (tmp_path / "profiles.db").exists()
        assert (tmp_path / "keyvault.json").exists()
        store.close()

    def test_initialize_twice_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        store.close()
        with pytest.raises(StoreAlreadyInitializedError):
            _store(tmp_path).initialize(_PASS)

    def test_unlock_before_initialize_raises(self, tmp_path: Path) -> None:
        with pytest.raises(StoreNotInitializedError):
            _store(tmp_path).unlock(_PASS)

    def test_weak_passphrase_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(WeakPassphraseError):
            _store(tmp_path).initialize("short")

    def test_operations_require_unlock(self, tmp_path: Path) -> None:
        store = _store(tmp_path)  # never initialized/unlocked
        with pytest.raises(DatabaseLockedError):
            store.add_person("Alice", consent_version="v1")

    def test_wrong_passphrase_fails_closed(self, tmp_path: Path) -> None:
        _store(tmp_path).initialize(_PASS)  # leaves files on disk
        with pytest.raises(InvalidPassphraseError):
            _store(tmp_path).unlock(_OTHER)

    def test_reopen_after_close_reads_data(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        person = store.add_person("Alice Example", consent_version="v1")
        store.close()

        reopened = _store(tmp_path)
        reopened.unlock(_PASS)
        loaded = reopened.get_person(person.id)
        assert loaded is not None
        assert loaded.display_name == "Alice Example"
        reopened.close()


class TestPeopleAndEmbeddings:
    def test_add_and_get_person_roundtrip(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        person = store.add_person("Bob", consent_version="v1")
        loaded = store.get_person(person.id)
        assert loaded is not None
        assert loaded.display_name == "Bob"
        assert loaded.active is True
        store.close()

    def test_get_missing_person_returns_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        assert store.get_person("nope") is None
        store.close()

    def test_add_embedding_roundtrip_preserves_vector(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        person = store.add_person("Bob", consent_version="v1")
        vector = np.arange(512, dtype=np.float32) / 512.0
        store.add_embedding(person.id, "face", "insightface/buffalo_l", vector, quality=0.9)

        loaded = store.get_embeddings(person.id)
        assert len(loaded) == 1
        assert loaded[0].kind == "face"
        assert loaded[0].vector.dtype == np.float32
        np.testing.assert_array_equal(loaded[0].vector, vector)
        store.close()

    def test_add_embedding_for_unknown_person_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        with pytest.raises(PersonNotFoundError):
            store.add_embedding("ghost", "face", "m", np.zeros(4, dtype=np.float32), 0.5)
        store.close()

    def test_list_people_excludes_inactive_by_default(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        keep = store.add_person("Keep", consent_version="v1")
        drop = store.add_person("Revoked", consent_version="v1")
        store.revoke_person(drop.id)

        active_ids = {p.id for p in store.list_people()}
        assert active_ids == {keep.id}
        all_ids = {p.id for p in store.list_people(include_inactive=True)}
        assert all_ids == {keep.id, drop.id}
        store.close()


class TestConsentAndDeletion:
    def test_consent_granted_recorded_on_add(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        person = store.add_person("Bob", consent_version="v1", consent_note="in person")
        events = store.consent_events(person.id)
        assert [e.event for e in events] == ["granted"]
        assert events[0].note == "in person"
        store.close()

    def test_delete_person_removes_person_and_embeddings(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        person = store.add_person("Bob", consent_version="v1")
        store.add_embedding(person.id, "face", "m", np.zeros(8, dtype=np.float32), 0.8)

        store.delete_person(person.id)
        assert store.get_person(person.id) is None
        assert store.get_embeddings(person.id) == []
        # The consent trail survives deletion (no biometric data in it).
        assert [e.event for e in store.consent_events(person.id)] == ["granted", "deleted"]
        store.close()

    def test_audit_log_records_actions(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        store.add_person("Bob", consent_version="v1")
        actions = {entry.action for entry in store.audit_entries()}
        assert "store_initialized" in actions
        assert "person_added" in actions
        store.close()


class TestPassphraseChange:
    def test_change_passphrase_rotates_access(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        person = store.add_person("Bob", consent_version="v1")
        store.change_passphrase(_PASS, _OTHER)
        store.close()

        # Old passphrase no longer works...
        with pytest.raises(InvalidPassphraseError):
            _store(tmp_path).unlock(_PASS)
        # ...new one does, and data is intact (envelope encryption: no re-encrypt).
        reopened = _store(tmp_path)
        reopened.unlock(_OTHER)
        loaded = reopened.get_person(person.id)
        assert loaded is not None
        assert loaded.display_name == "Bob"
        reopened.close()

    def test_change_passphrase_wrong_current_raises(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        with pytest.raises(InvalidPassphraseError):
            store.change_passphrase(_OTHER, "brand-new-passphrase")
        store.close()


class TestEncryptionAtRest:
    def test_plaintext_name_absent_from_database_file(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        store.add_person("Alice Secret Name", consent_version="v1")
        store.close()

        raw = (tmp_path / "profiles.db").read_bytes()
        assert b"Alice Secret Name" not in raw

    def test_embedding_bytes_absent_from_database_file(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.initialize(_PASS)
        person = store.add_person("Bob", consent_version="v1")
        vector = np.linspace(0, 1, 128, dtype=np.float32)
        store.add_embedding(person.id, "face", "m", vector, 0.9)
        store.close()

        raw = (tmp_path / "profiles.db").read_bytes()
        assert vector.tobytes() not in raw
