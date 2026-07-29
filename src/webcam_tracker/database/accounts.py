"""Multi-user login layer for the identity store (Stage 2.5).

Sits ON TOP of the same envelope-encryption scheme as crypto.py / ProfileStore,
adding named accounts with per-user passphrases in two selectable modes:

  * "shared"   -- one shared passphrase unlocks a single store-wide DEK. Names
                  are unique login labels; every logged-in user can see/track
                  everyone (the DEK decrypts all data). Joining needs the shared
                  passphrase -- that's the "shared key".
  * "per_user" -- each account has its OWN passphrase wrapping its OWN DEK, so a
                  login only decrypts that user's data. Anyone can enroll a fresh
                  account without knowing anyone else's secret, and a user who
                  forgets their passphrase can delete just their own account and
                  start over.

The account file (accounts.json, safe in the clear) holds only salts, KDF params
and wrapped DEKs -- never a passphrase or a raw DEK. Raw names aren't stored
either: only a per-store salted hash (`hash_name`), which is enough to enforce
uniqueness, look up a login, and delete-by-name, all without a passphrase.

The mode is fixed when the store is created (the two on-disk shapes are not
interchangeable) and reported by `mode` afterwards.
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from webcam_tracker.database.crypto import (
    KdfParams,
    hash_name,
    new_dek,
    new_name_salt,
    unwrap_dek,
    wrap_dek,
)
from webcam_tracker.database.errors import WeakPassphraseError
from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)

_ACCOUNTS_VERSION = 2
MODE_SHARED = "shared"
MODE_PER_USER = "per_user"
_MODES = (MODE_SHARED, MODE_PER_USER)


class AccountError(Exception):
    """Base class for account-layer errors."""


class NameTakenError(AccountError):
    """The requested display name is already registered (names are unique)."""


class NoSuchAccountError(AccountError):
    """No account is registered under that name."""


class StoreModeError(AccountError):
    """An operation isn't valid for the store's current mode."""


@dataclass(frozen=True)
class Login:
    """Result of a successful create/login/enroll: the unlocked data key plus
    who/what it belongs to.

    `scope_person_id` is the person_id the caller should restrict the data store
    to. It is None in shared mode (full access) and the account's user_id in
    per_user mode (only that user's own data is readable)."""

    user_id: str
    display_name: str
    dek: bytes
    mode: str
    scope_person_id: str | None


class AccountManager:
    """Reads/writes accounts.json and turns (name, passphrase) into an unlocked
    DEK. Knows nothing about faces or SQLite -- it only manages login secrets."""

    def __init__(
        self, accounts_path: str | Path, kdf_params: KdfParams, min_passphrase_length: int
    ) -> None:
        self._path = Path(accounts_path)
        self._kdf = kdf_params
        self._min_len = min_passphrase_length

    # ----------------------------------------------------------------- state

    def exists(self) -> bool:
        """True if a store has been created (the account file exists)."""
        return self._path.exists()

    def destroy(self) -> None:
        """Delete the whole account file, un-committing the mode. Used to fully
        roll back a failed FIRST enrollment (so the store isn't left locked into
        a mode with no usable accounts)."""
        self._path.unlink(missing_ok=True)

    @property
    def mode(self) -> str:
        """The mode this store was created in. Raises if no store exists yet."""
        return str(self._read()["mode"])

    def has_accounts(self) -> bool:
        """True if at least one account is registered."""
        return self.exists() and bool(self._names(self._read()))

    def has_name(self, name: str) -> bool:
        """True if `name` is already registered (case-insensitive)."""
        if not self.exists():
            return False
        data = self._read()
        return hash_name(name, self._name_salt(data)) in self._names(data)

    # ---------------------------------------------------------------- create

    def create(self, mode: str, name: str, passphrase: str) -> Login:
        """Create a brand-new store in `mode` with its first account. The
        passphrase given here becomes: the shared key (shared mode) or this
        user's own key (per_user mode)."""
        if self.exists():
            raise AccountError(f"a store already exists at {self._path}")
        if mode not in _MODES:
            raise StoreModeError(f"unknown mode {mode!r}; use one of {_MODES}")
        self._check_strength(passphrase)

        user_id = uuid.uuid4().hex
        name_salt = new_name_salt()
        dek = new_dek()
        data: dict[str, Any] = {
            "version": _ACCOUNTS_VERSION,
            "mode": mode,
            "kdf_params": self._kdf_dict(),
            "name_salt": name_salt,
        }
        name_hash = hash_name(name, self._decode_salt(name_salt))
        envelope = wrap_dek(dek, passphrase, self._kdf)
        if mode == MODE_SHARED:
            # One shared envelope + a set of registered name-hashes.
            data["keyvault"] = envelope
            data["names"] = {name_hash: user_id}
        else:
            # One envelope per account, each wrapping that account's own DEK.
            data["accounts"] = {name_hash: {"user_id": user_id, **envelope}}
        self._write(data)
        logger.info("Identity store created", extra={"mode": mode})
        return self._login_result(user_id, name, dek, mode)

    # ----------------------------------------------------------------- login

    def login(self, name: str, passphrase: str) -> Login:
        """Authenticate an existing account. Raises NoSuchAccountError if the
        name isn't registered, InvalidPassphraseError if the passphrase is
        wrong."""
        data = self._require_store()
        mode = str(data["mode"])
        name_hash = hash_name(name, self._name_salt(data))
        if mode == MODE_SHARED:
            names = self._names(data)
            if name_hash not in names:
                raise NoSuchAccountError(name)
            dek = unwrap_dek(data["keyvault"], passphrase, self._kdf)  # raises if wrong
            return self._login_result(names[name_hash], name, dek, mode)
        # per_user
        account = data["accounts"].get(name_hash)
        if account is None:
            raise NoSuchAccountError(name)
        dek = unwrap_dek(account, passphrase, self._kdf)  # raises if wrong
        return self._login_result(account["user_id"], name, dek, mode)

    # ---------------------------------------------------------------- enroll

    def enroll(self, name: str, passphrase: str) -> Login:
        """Add a NEW account to an existing store and return its unlocked DEK.

        * per_user: creates a fresh DEK for this user; no other secret needed.
        * shared:   the passphrase MUST be the store's shared key (it has to
                    unwrap the shared DEK) -- that's how a new member joins.

        Raises NameTakenError if the name is in use; in shared mode raises
        InvalidPassphraseError if the passphrase isn't the shared key."""
        data = self._require_store()
        mode = str(data["mode"])
        self._check_strength(passphrase)
        name_hash = hash_name(name, self._name_salt(data))
        if self._name_registered(data, name_hash):
            raise NameTakenError(name)

        user_id = uuid.uuid4().hex
        if mode == MODE_SHARED:
            # Prove knowledge of the shared key by unwrapping the shared DEK,
            # then just record the new unique name against the same store.
            dek = unwrap_dek(data["keyvault"], passphrase, self._kdf)  # raises if wrong
            data["names"][name_hash] = user_id
        else:
            dek = new_dek()
            data["accounts"][name_hash] = {
                "user_id": user_id,
                **wrap_dek(dek, passphrase, self._kdf),
            }
        self._write(data)
        logger.info("Account enrolled", extra={"mode": mode})
        return self._login_result(user_id, name, dek, mode)

    # ---------------------------------------------------- forgot / delete

    def delete_account(self, name: str) -> str:
        """Remove an account by name WITHOUT its passphrase (the "I forgot"
        path). Returns the user_id so the caller can purge that person's data.

        In shared mode this only frees the name for re-use; the shared DEK and
        everyone else's access are unaffected. In per_user mode it discards the
        account's envelope, permanently orphaning that user's encrypted data
        (which the caller should purge)."""
        data = self._require_store()
        mode = str(data["mode"])
        name_hash = hash_name(name, self._name_salt(data))
        if not self._name_registered(data, name_hash):
            raise NoSuchAccountError(name)
        if mode == MODE_SHARED:
            user_id = str(data["names"].pop(name_hash))
        else:
            user_id = str(data["accounts"].pop(name_hash)["user_id"])
        self._write(data)
        logger.info("Account deleted", extra={"mode": mode})
        return user_id

    # -------------------------------------------------- change passphrase

    def change_passphrase(self, name: str, current: str, new: str) -> None:
        """Change one account's passphrase.

        In per_user mode this re-wraps that user's own DEK. In shared mode the
        passphrase IS the shared key, so changing it re-wraps the store-wide
        DEK for everyone -- every other user must then use the new key too."""
        data = self._require_store()
        mode = str(data["mode"])
        self._check_strength(new)
        name_hash = hash_name(name, self._name_salt(data))
        if mode == MODE_SHARED:
            if name_hash not in self._names(data):
                raise NoSuchAccountError(name)
            dek = unwrap_dek(data["keyvault"], current, self._kdf)  # raises if wrong
            data["keyvault"] = wrap_dek(dek, new, self._kdf)
        else:
            account = data["accounts"].get(name_hash)
            if account is None:
                raise NoSuchAccountError(name)
            dek = unwrap_dek(account, current, self._kdf)  # raises if wrong
            data["accounts"][name_hash] = {
                "user_id": account["user_id"],
                **wrap_dek(dek, new, self._kdf),
            }
        self._write(data)

    # -------------------------------------------------------------- internals

    def _login_result(self, user_id: str, name: str, dek: bytes, mode: str) -> Login:
        scope = None if mode == MODE_SHARED else user_id
        return Login(user_id=user_id, display_name=name, dek=dek, mode=mode, scope_person_id=scope)

    def _name_registered(self, data: dict[str, Any], name_hash: str) -> bool:
        return name_hash in self._names(data)

    @staticmethod
    def _names(data: dict[str, Any]) -> dict[str, str]:
        """Map of registered name_hash -> user_id, uniform across both modes."""
        if data["mode"] == MODE_SHARED:
            names: dict[str, str] = data.get("names", {})
            return names
        return {h: acct["user_id"] for h, acct in data.get("accounts", {}).items()}

    @staticmethod
    def _name_salt(data: dict[str, Any]) -> bytes:
        return base64.b64decode(data["name_salt"])

    @staticmethod
    def _decode_salt(name_salt_b64: str) -> bytes:
        return base64.b64decode(name_salt_b64)

    def _kdf_dict(self) -> dict[str, int]:
        return {
            "time_cost": self._kdf.time_cost,
            "memory_kib": self._kdf.memory_kib,
            "parallelism": self._kdf.parallelism,
        }

    def _check_strength(self, passphrase: str) -> None:
        if len(passphrase) < self._min_len:
            raise WeakPassphraseError(f"passphrase must be at least {self._min_len} characters")

    def _require_store(self) -> dict[str, Any]:
        if not self.exists():
            raise AccountError(f"no store at {self._path}; create one first")
        return self._read()

    def _read(self) -> dict[str, Any]:
        with self._path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)
            return data

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)


__all__ = [
    "AccountError",
    "AccountManager",
    "Login",
    "MODE_PER_USER",
    "MODE_SHARED",
    "NameTakenError",
    "NoSuchAccountError",
    "StoreModeError",
]
