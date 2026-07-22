"""The encrypted identity profile store (Stage 2.1).

A SQLite-backed store for registered people + their biometric embeddings +
consent metadata, with sensitive columns encrypted at the application layer
(AES-256-GCM under a passphrase-derived key -- see crypto.py and
docs/database_design.md).

Lifecycle: construct -> `initialize(passphrase)` once (first run) OR
`unlock(passphrase)` (subsequent runs) -> use -> `close()`. Data operations
before an unlock raise DatabaseLockedError. Usable as a context manager.

Not thread-safe: one store instance is meant to be driven from one thread
(the pipeline loop / a CLI), matching how the rest of the project is used.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from webcam_tracker.database.crypto import (
    KdfParams,
    create_keyvault,
    decrypt_field,
    encrypt_field,
    rewrap_dek,
    unlock_keyvault,
)
from webcam_tracker.database.errors import (
    DatabaseLockedError,
    PersonNotFoundError,
    StoreAlreadyInitializedError,
    StoreNotInitializedError,
    WeakPassphraseError,
)
from webcam_tracker.database.models import AuditEntry, ConsentEvent, Embedding, Person
from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)

_SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProfileStore:
    def __init__(
        self,
        db_path: str | Path,
        keyvault_path: str | Path,
        kdf_params: KdfParams,
        min_passphrase_length: int,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._db_path = Path(db_path)
        self._keyvault_path = Path(keyvault_path)
        self._kdf_params = kdf_params
        self._min_passphrase_length = min_passphrase_length
        self._now = now
        self._conn: sqlite3.Connection | None = None
        self._dek: bytes | None = None

    # ------------------------------------------------------------------ lifecycle

    def is_initialized(self) -> bool:
        """True if this store has been set up (its key vault exists)."""
        return self._keyvault_path.exists()

    def initialize(self, passphrase: str) -> None:
        """Create a brand-new encrypted store. Errors if one already exists."""
        if self.is_initialized():
            raise StoreAlreadyInitializedError(
                f"identity store already initialized at {self._keyvault_path}"
            )
        self._check_passphrase_strength(passphrase)
        self._keyvault_path.parent.mkdir(parents=True, exist_ok=True)
        keyvault, dek = create_keyvault(passphrase, self._kdf_params)
        self._write_keyvault(keyvault)
        self._dek = dek
        self._conn = self._connect()
        self._create_schema()
        self._audit("store_initialized", "")
        self._conn.commit()
        logger.info("Identity store initialized", extra={"path": str(self._db_path)})

    def unlock(self, passphrase: str) -> None:
        """Open an existing store. Raises InvalidPassphraseError on a wrong
        passphrase (fails closed -- no partial/garbage unlock)."""
        if not self.is_initialized():
            raise StoreNotInitializedError(
                f"no identity store at {self._keyvault_path}; call initialize() first"
            )
        keyvault = self._read_keyvault()
        self._dek = unlock_keyvault(keyvault, passphrase)  # raises InvalidPassphraseError
        self._conn = self._connect()
        logger.info("Identity store unlocked", extra={"path": str(self._db_path)})

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._dek = None

    def __enter__(self) -> ProfileStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def change_passphrase(self, current_passphrase: str, new_passphrase: str) -> None:
        """Re-wrap the data key under a new passphrase. Does NOT re-encrypt the
        data (envelope encryption). Verifies the current passphrase first."""
        conn, dek = self._require_unlocked()
        self._check_passphrase_strength(new_passphrase)
        unlock_keyvault(self._read_keyvault(), current_passphrase)  # raises if wrong
        self._write_keyvault(rewrap_dek(dek, new_passphrase, self._kdf_params))
        self._audit("passphrase_changed", "")
        conn.commit()

    # -------------------------------------------------------------------- people

    def add_person(self, display_name: str, consent_version: str, consent_note: str = "") -> Person:
        """Register a person, recording a 'granted' consent event."""
        conn, dek = self._require_unlocked()
        person_id = uuid.uuid4().hex
        created_at = self._now().isoformat()
        conn.execute(
            "INSERT INTO person (id, display_name, created_at, active, consent_version) "
            "VALUES (?, ?, ?, 1, ?)",
            (
                person_id,
                encrypt_field(dek, display_name.encode("utf-8")),
                created_at,
                consent_version,
            ),
        )
        self._insert_consent(person_id, "granted", consent_note, created_at)
        self._audit("person_added", person_id)
        conn.commit()
        return Person(
            id=person_id,
            display_name=display_name,
            created_at=created_at,
            active=True,
            consent_version=consent_version,
        )

    def get_person(self, person_id: str) -> Person | None:
        conn, dek = self._require_unlocked()
        row = conn.execute("SELECT * FROM person WHERE id = ?", (person_id,)).fetchone()
        return self._row_to_person(row, dek) if row is not None else None

    def list_people(self, include_inactive: bool = False) -> list[Person]:
        conn, dek = self._require_unlocked()
        query = "SELECT * FROM person"
        if not include_inactive:
            query += " WHERE active = 1"
        query += " ORDER BY created_at"
        return [self._row_to_person(row, dek) for row in conn.execute(query).fetchall()]

    def revoke_person(self, person_id: str, note: str = "") -> None:
        """Mark a person inactive (stops matching) without deleting their data."""
        conn, _ = self._require_unlocked()
        self._ensure_person_exists(person_id)
        timestamp = self._now().isoformat()
        conn.execute("UPDATE person SET active = 0 WHERE id = ?", (person_id,))
        self._insert_consent(person_id, "revoked", note, timestamp)
        self._audit("person_revoked", person_id)
        conn.commit()

    def delete_person(self, person_id: str, note: str = "") -> None:
        """Hard-delete a person and their embeddings. Keeps the consent/audit
        trail (which holds no biometric data). See docs/database_design.md sec
        5.4 for the backup caveat."""
        conn, _ = self._require_unlocked()
        self._ensure_person_exists(person_id)
        timestamp = self._now().isoformat()
        self._insert_consent(person_id, "deleted", note, timestamp)
        conn.execute("DELETE FROM person WHERE id = ?", (person_id,))  # cascades to embeddings
        self._audit("person_deleted", person_id)
        conn.commit()
        conn.execute("VACUUM")  # reclaim/overwrite the freed pages

    # ---------------------------------------------------------------- embeddings

    def add_embedding(
        self,
        person_id: str,
        kind: str,
        model_id: str,
        vector: np.ndarray,
        quality: float,
    ) -> Embedding:
        conn, dek = self._require_unlocked()
        self._ensure_person_exists(person_id)
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        embedding_id = uuid.uuid4().hex
        created_at = self._now().isoformat()
        conn.execute(
            "INSERT INTO embedding "
            "(id, person_id, kind, model_id, dim, vector, quality, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                embedding_id,
                person_id,
                kind,
                model_id,
                int(vec.size),
                encrypt_field(dek, vec.tobytes()),
                float(quality),
                created_at,
            ),
        )
        self._audit("embedding_added", f"{person_id}:{kind}")
        conn.commit()
        return Embedding(
            id=embedding_id,
            person_id=person_id,
            kind=kind,
            model_id=model_id,
            vector=vec,
            quality=quality,
            created_at=created_at,
        )

    def get_embeddings(self, person_id: str, kind: str | None = None) -> list[Embedding]:
        conn, dek = self._require_unlocked()
        query = "SELECT * FROM embedding WHERE person_id = ?"
        params: list[Any] = [person_id]
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY created_at"
        return [self._row_to_embedding(row, dek) for row in conn.execute(query, params).fetchall()]

    # ------------------------------------------------------------ consent / audit

    def consent_events(self, person_id: str) -> list[ConsentEvent]:
        conn, _ = self._require_unlocked()
        rows = conn.execute(
            "SELECT * FROM consent_event WHERE person_id = ? ORDER BY timestamp", (person_id,)
        ).fetchall()
        return [
            ConsentEvent(
                id=row["id"],
                person_id=row["person_id"],
                event=row["event"],
                timestamp=row["timestamp"],
                note=row["note"],
            )
            for row in rows
        ]

    def audit(self, action: str, detail: str = "") -> None:
        """Record an audit-log entry and commit it. For external callers (e.g.
        the state machine logging identity decisions)."""
        conn, _ = self._require_unlocked()
        self._audit(action, detail)
        conn.commit()

    def audit_entries(self, limit: int = 100) -> list[AuditEntry]:
        conn, _ = self._require_unlocked()
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            AuditEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                action=row["action"],
                detail=row["detail"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------- internals

    def _require_unlocked(self) -> tuple[sqlite3.Connection, bytes]:
        if self._conn is None or self._dek is None:
            raise DatabaseLockedError("identity store is locked; call initialize() or unlock()")
        return self._conn, self._dek

    def _ensure_person_exists(self, person_id: str) -> None:
        conn, _ = self._require_unlocked()
        row = conn.execute("SELECT 1 FROM person WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            raise PersonNotFoundError(person_id)

    def _check_passphrase_strength(self, passphrase: str) -> None:
        if len(passphrase) < self._min_passphrase_length:
            raise WeakPassphraseError(
                f"passphrase must be at least {self._min_passphrase_length} characters"
            )

    def _row_to_person(self, row: sqlite3.Row, dek: bytes) -> Person:
        return Person(
            id=row["id"],
            display_name=decrypt_field(dek, row["display_name"]).decode("utf-8"),
            created_at=row["created_at"],
            active=bool(row["active"]),
            consent_version=row["consent_version"],
        )

    def _row_to_embedding(self, row: sqlite3.Row, dek: bytes) -> Embedding:
        raw = decrypt_field(dek, row["vector"])
        vector = np.frombuffer(raw, dtype=np.float32).copy()  # copy -> writable, owns memory
        return Embedding(
            id=row["id"],
            person_id=row["person_id"],
            kind=row["kind"],
            model_id=row["model_id"],
            vector=vector,
            quality=row["quality"],
            created_at=row["created_at"],
        )

    def _insert_consent(self, person_id: str, event: str, note: str, timestamp: str) -> None:
        conn, _ = self._require_unlocked()
        conn.execute(
            "INSERT INTO consent_event (id, person_id, event, timestamp, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, person_id, event, timestamp, note),
        )

    def _audit(self, action: str, detail: str) -> None:
        conn, _ = self._require_unlocked()
        conn.execute(
            "INSERT INTO audit_log (id, timestamp, action, detail) VALUES (?, ?, ?, ?)",
            (uuid.uuid4().hex, self._now().isoformat(), action, detail),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                schema_version INTEGER NOT NULL,
                created_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS person (
                id              TEXT PRIMARY KEY,
                display_name    BLOB NOT NULL,
                created_at      TEXT NOT NULL,
                active          INTEGER NOT NULL,
                consent_version TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding (
                id         TEXT PRIMARY KEY,
                person_id  TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
                kind       TEXT NOT NULL,
                model_id   TEXT NOT NULL,
                dim        INTEGER NOT NULL,
                vector     BLOB NOT NULL,
                quality    REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embedding_person ON embedding(person_id);
            CREATE TABLE IF NOT EXISTS consent_event (
                id        TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                event     TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                note      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id        TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                action    TEXT NOT NULL,
                detail    TEXT NOT NULL
            );
            """
        )
        if self._conn.execute("SELECT COUNT(*) FROM schema_meta").fetchone()[0] == 0:
            self._conn.execute(
                "INSERT INTO schema_meta (schema_version, created_at) VALUES (?, ?)",
                (_SCHEMA_VERSION, self._now().isoformat()),
            )

    def _read_keyvault(self) -> dict[str, Any]:
        with self._keyvault_path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)
            return data

    def _write_keyvault(self, keyvault: dict[str, Any]) -> None:
        with self._keyvault_path.open("w", encoding="utf-8") as handle:
            json.dump(keyvault, handle, indent=2)
