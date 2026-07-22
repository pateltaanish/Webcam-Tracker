# Stage 2 — Registration, Identity & Encrypted Database Design

**Status: design doc (no code yet), written before Stage 2.1 per the project's
spec-before-code workflow.** This is the foundational design for the part of
the system that makes it *consent-based person identification* rather than
generic "any person" tracking. Because it handles **biometric data**, the
design leads with consent, data minimization, and encryption — those are
requirements, not afterthoughts.

Confirmed decisions (from Taanish, 2026-07-22):

- **Encryption key = operator passphrase.** Derived at startup via a KDF,
  never written to disk. The store is useless without it.
- **Face model = InsightFace (SCRFD detector + ArcFace embeddings).** Matches
  `docs/02_architecture.md` §3 and the eventual embedded target.
- **Build sequencing = face-first vertical slice** (recommended and accepted):
  database → registration → face recognition → identity fusion end to end
  first, *then* add body Re-ID (OSNet). This gets a working "lock onto the
  registered person" result sooner and lets us validate the hardest part
  (identity) before broadening it.

---

## 1. Consent model & non-goals (the ethical core)

This system only ever tracks a person who has **knowingly registered and
consented**. Restating the spec's non-goals concretely for this stage:

- **No enrollment without the subject present and consenting.** Registration
  is an interactive, in-person flow (Stage 2.2). There is no way to enroll
  someone from a photo, a video, or a scraped image — the flow requires a live
  capture session the subject participates in.
- **No covert or stranger identification.** An unregistered person always
  resolves to the explicit outcome `UNKNOWN`. The system never tries to name,
  profile, or track someone who isn't enrolled.
- **A consent record is stored with every profile** and every enrollment,
  revocation, and deletion is written to an audit log (§6).
- **The subject can revoke and be deleted at any time**, and deletion is
  designed to be cryptographically final (§5.4), not a soft flag.
- **Local only.** Nothing leaves the machine — no cloud, no network calls for
  identity. (The only network access in the whole project is one-time model
  weight downloads.)

If a future change would weaken any of the above, it needs an explicit
decision, not a silent code change.

## 2. Threat model

What this design **does** protect against:

- **Theft of the data at rest** (laptop/SD-card stolen, disk imaged, file
  copied off the machine). Without the operator passphrase, the profile store
  is an opaque encrypted blob.
- **Casual inspection** — no plaintext names or embeddings sitting in a file.

What it **does not** (and cannot, at this stage) protect against:

- **A compromised running machine.** Once the operator unlocks the store, the
  data-encryption key and decrypted embeddings live in process memory. Malware
  running as the operator, or a memory dump of the live process, can read them.
  Mitigating that is an OS-hardening problem, out of scope here.
- **A weak passphrase.** The KDF (§5.1) raises the cost of brute force, but a
  guessable passphrase is still the weakest link. The registration/startup
  flow will warn on short passphrases.
- **Coercion of the operator.** Not a technical problem.

This is stated plainly so we don't oversell the guarantee: **encryption at
rest, gated by an operator passphrase** — solid against a lost device, not a
defense against a live compromise.

## 3. What is stored (data minimization)

The guiding rule: **store the minimum needed to recognize an enrolled person,
and nothing that isn't needed.**

| Data | Stored? | Notes |
|---|---|---|
| Face **embeddings** (512-d ArcFace vectors) | **Yes, encrypted** | The core biometric. Not human-viewable; still treated as sensitive biometric data. |
| Body **Re-ID embeddings** (later slice) | Yes, encrypted | Added when Re-ID lands. |
| Display name / label | Yes, encrypted | So the operator can tell profiles apart. Can be a pseudonym. |
| Consent metadata | Yes | Consent version, timestamp, who enrolled. |
| **Raw face images / video** | **No, by default** | We store *embeddings*, not photos. A config flag can retain a few sample crops for debugging, **off by default and clearly consent-gated** — do not enable casually. |
| Anything else about the person | No | No age/gender/notes fields tempting scope creep. |

Rationale for not storing raw images: an embedding is far less directly
identifying/abusable than a face photo, and not keeping photos is the single
biggest privacy win available. (Embeddings are *not* trivially reversible to a
face, but they are still biometric data and are encrypted regardless.)

## 4. Storage backend

**SQLite** (Python stdlib `sqlite3`), a single file under
`data/identity/profiles.db` (the `data/` tree is gitignored). Sensitive
columns are encrypted at the **application layer** (§5) — we do *not* rely on
SQLCipher, to avoid a native build that's painful on Windows and on the future
embedded target.

Why SQLite over flat files: transactional integrity (a half-written profile
can't corrupt the store), easy relational queries (a person's N embeddings +
consent events), and a single portable file that's trivial to back up or
delete.

Proposed schema (v1):

```
schema_meta(       -- one row; plaintext, non-sensitive
  schema_version, created_at )

person(
  id            TEXT PK,     -- random UUID, not sequential
  display_name  BLOB,       -- encrypted
  created_at    TEXT,
  active        INTEGER,     -- 0 after revoke/delete
  consent_version TEXT )

embedding(
  id         TEXT PK,
  person_id  TEXT FK -> person(id) ON DELETE CASCADE,
  kind       TEXT,          -- 'face' | 'reid'
  model_id   TEXT,          -- e.g. 'insightface/buffalo_l/arcface-r100'
  dim        INTEGER,       -- 512 for ArcFace
  vector     BLOB,          -- encrypted float32[dim]
  quality    REAL,          -- capture quality score (§ registration)
  created_at TEXT )

consent_event(
  id        TEXT PK,
  person_id TEXT,           -- kept even after person delete, for the audit trail
  event     TEXT,           -- 'granted' | 'revoked' | 'deleted'
  timestamp TEXT,
  note      TEXT )

audit_log(                  -- see §6
  id, timestamp, action, detail )
```

The crypto header (salt, KDF params, wrapped data key) lives in a **separate
sidecar file**, `data/identity/keyvault.json` (plaintext JSON — its contents
are safe to store in the clear; see §5). Keeping it separate from the `.db`
makes "change passphrase" and "verify unlock" simple and keeps key material
out of the data file.

## 5. Encryption design

Envelope encryption with an operator passphrase. Two keys:

- **KEK** (key-encryption key): derived from the operator passphrase — never
  stored.
- **DEK** (data-encryption key): a random 256-bit key that actually encrypts
  the data. Stored only in *wrapped* (KEK-encrypted) form.

### 5.1 Deriving the KEK from the passphrase

- KDF: **Argon2id** (memory-hard, resists GPU brute force), via
  `argon2-cffi`. Parameters tuned to ~0.5–1s on the dev machine (e.g. 64 MiB
  memory, time-cost 3, parallelism 4) — recorded in `keyvault.json` so they
  can be raised later without breaking existing stores.
- A random 16-byte **salt** (stored in `keyvault.json`, plaintext — salts are
  not secret) makes precomputation/rainbow tables useless.
- Output: a 256-bit KEK, held in memory only for as long as needed to unwrap
  the DEK.

### 5.2 The data-encryption key (DEK)

- On first setup: generate a random 256-bit DEK. Encrypt ("wrap") it with the
  KEK using **AES-256-GCM** (authenticated). Store `{salt, kdf_params, nonce,
  wrapped_dek}` in `keyvault.json`.
- On startup: operator enters passphrase → derive KEK → unwrap DEK. A wrong
  passphrase fails the GCM auth check immediately (no partial/garbage unlock),
  which is how we validate the passphrase without storing a hash of it.
- **Changing the passphrase** re-derives a new KEK and re-wraps the *same*
  DEK — no need to re-encrypt the whole database.

### 5.3 Encrypting the data

- Every sensitive field (`display_name`, `embedding.vector`) is encrypted with
  the DEK using **AES-256-GCM**, a **fresh random 96-bit nonce per value**,
  storing `nonce || ciphertext || tag` in the BLOB.
- Library: the `cryptography` package (`AESGCM`) — well-maintained, ships
  Windows wheels, no native build step.
- GCM gives us *authenticated* encryption: tampering with a stored ciphertext
  is detected on decrypt, not silently accepted.

### 5.4 Deletion / crypto-shredding

- Normal delete: `DELETE` the person row (cascades to embeddings) + `VACUUM`
  to reclaim/overwrite pages. A `consent_event('deleted')` and `audit_log`
  entry remain (they contain no biometric data).
- Caveat stated honestly: `VACUUM` + row delete makes recovery from the live
  file impractical, but **old backups still hold the data**. For a stronger
  "make it unrecoverable everywhere" guarantee, a future option is
  **per-profile DEKs** (each person's data under its own key, wrapped by the
  master DEK): deleting a person's key crypto-shreds them even in backups.
  Deferred as a v2 enhancement — v1 uses a single DEK for simplicity, with the
  backup caveat documented.

## 6. Audit log

Every enrollment, revocation, deletion, passphrase change, and — importantly —
every **identity decision the state machine acts on** (locked onto person X,
rejected an unknown, lost/reacquired X) is written to `audit_log` with a
timestamp. This doubles as the debugging trail for false-lock / identity-switch
incidents called for in `docs/02_architecture.md` §4. The audit log stores
person **ids** (UUIDs) and actions, **not** embeddings or images.

## 7. Face recognition pipeline (the vertical slice)

InsightFace `FaceAnalysis` (SCRFD detection + ArcFace `buffalo_l`), run through
**onnxruntime** (GPU via `onnxruntime-gpu` on the RTX 3060, CPU fallback):

1. Detect faces in the frame (SCRFD) → face boxes + 5-point landmarks.
2. Align + embed each face (ArcFace) → **512-d embedding, L2-normalized**.
3. Compare to enrolled profiles by **cosine similarity** (= dot product on
   normalized vectors). A person may have several stored embeddings (multiple
   angles); score against the best / an aggregate.
4. Decision: best similarity ≥ `match_threshold` → that person; otherwise
   `UNKNOWN`. Threshold is a config value, tuned on Taanish's own captured data
   in Stage 2.5 (FAR/FRR), not guessed permanently.

**Licensing flag (do not skip):** InsightFace *code* is MIT, but the
pretrained `buffalo_l` **models are released for non-commercial / research
use**. This is the same class of caveat as YOLO11n's AGPL (see
`docs/model_licenses.md`) and must be recorded there before shipping. If the
project ever goes commercial, the face model needs revisiting. Calling it out
now so it isn't discovered late.

## 8. Identity fusion & how it plugs into Stage 1

`identity` (Stage 2.4) combines, per tracked person:

- face match similarity (when a face is visible and embeddable),
- Re-ID similarity (later slice, for when the face is turned away/occluded),
- temporal consistency (this track has matched person X for the last N frames),

into a single **confidence score**. That score is exactly what the state
machine's deferred **`CANDIDATE_DETECTED` → `VERIFYING_IDENTITY` → `TRACKING`**
path (already reserved in `state_machine.py`) consumes. Two concrete Stage-1
placeholders get replaced:

- **Selection**: instead of clicking a track, the operator selects a
  *registered person*; the system finds which live track (if any) is them.
- **Reacquisition**: the identity-free "grab the nearest returning track"
  becomes **identity-gated** — a returning candidate is only re-locked if its
  face/Re-ID confidence clears the threshold. This is the real fix for the
  "leaves and comes back as a new track ID" problem: the new track id is
  matched back to the same *person*, not just the same *place*.

The `TargetSelector`/`state_machine` interfaces were built in Stage 1
specifically so this swap doesn't disturb anything downstream (gimbal,
recovery, drone-control signal).

## 9. New dependencies

Pinned in a new `requirements-identity.txt` (kept separate from
`requirements-ml.txt` like the other heavy optional group):

- `insightface`, `onnxruntime-gpu` (CPU `onnxruntime` fallback) — face model
- `cryptography` — AES-256-GCM
- `argon2-cffi` — Argon2id KDF

Windows/onnxruntime-gpu + CUDA version alignment is a known fiddly spot (same
family of issue as the torch CUDA-wheel dance in Stage 1.3) — expect a
verification pass on first install.

## 10. Build plan (maps to roadmap §2)

Face-first vertical slice, each step runnable/testable before the next:

- **2.1 `database`**: encrypted store + schema + passphrase unlock + CRUD +
  audit log. Unit-tested with a temp store and a throwaway passphrase; verify
  wrong-passphrase fails closed, delete removes data, round-trip encrypt/
  decrypt is lossless.
- **2.2 Registration CLI**: capture samples (live), quality gates
  (blur/exposure/single-face/size), multi-angle prompts, embed, store with a
  consent record. Verified by enrolling Taanish and inspecting the store.
- **2.3 `face_recognition`**: embedding extraction + cosine match against the
  store, with an explicit `UNKNOWN` outcome. Verified: enrolled person matches,
  a different person / empty frame → UNKNOWN.
- **2.4 `identity` fusion**: face + temporal consistency → confidence; wire
  into the state machine's VERIFYING_IDENTITY path; replace placeholder
  reacquisition with identity-gated reacquisition.
- **(then) Re-ID slice**: `reid` (OSNet) embeddings added to the same store and
  fusion, for face-away robustness.
- **2.5 Test harness**: known-target vs. unregistered vs. wrong-registered-user
  on Taanish's own captured data; measure FAR/FRR; tune thresholds.

## 11. Open items to confirm before/while building 2.1

- **Argon2id cost parameters** — pick after a quick benchmark on the dev
  machine (target ~0.5–1s unlock).
- **`match_threshold`** — starts as a documented placeholder; tuned for real in
  2.5, never hardcoded as "final."
- **Model license** — add the InsightFace `buffalo_l` non-commercial caveat to
  `docs/model_licenses.md` as part of 2.3.
- **Per-profile DEK crypto-shredding** — deferred to v2; revisit if the backup
  caveat in §5.4 matters for the real use case.
