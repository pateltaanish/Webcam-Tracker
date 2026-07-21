# Engineering Specification — Consent-Based Person Tracking Drone

Status: **Draft v1** (locked after clarification round with project owner, 2026-07-20)

## 1. Purpose

A ≤250 g takeoff-weight drone that detects, identifies, and tracks **one specific,
previously registered, consenting person**, keeping them centered in frame via
gimbal (and eventually flight) commands. The system is explicitly **not** designed
for identifying unregistered strangers or surveillance of public spaces.

## 2. Confirmed constraints (from clarification round)

| Parameter | Value |
|---|---|
| Dev machine | Windows 11, NVIDIA RTX 3090 (24 GB VRAM) — ample for Stage 1/2 training & inference |
| Developer CV/Python experience | Comfortable with Python; new to computer vision — all CV/ML concepts must be explained, not assumed |
| Existing hardware | None — full build from scratch |
| Operating environment | Indoor (well-lit) **and** outdoor daytime. Low-light is **not** an initial requirement (deferred / stretch goal) |
| Scene scale | 1–3 people in frame, close range (1–5 m) |
| Target perception rate | 30 FPS, low latency (stretch target — see Risk R2) |
| Region | United States → sub-250 g exempts this aircraft from FAA registration/Part 107 remote-ID hardware requirements for most operations, but Part 107 operating rules (or recreational rules under 49 U.S.C. §44809) still apply if flown outdoors. This is not legal advice; confirm current FAA rules before outdoor flights. |
| Frame | Custom 3D-printed CAD, effectively free (material cost only) |
| Motors/props | Inexpensive commodity micro brushless (budget: <$30 total) |
| Onboard-compute + camera + gimbal budget | **~$350 total** — this is the binding hardware constraint (see Risk R1) |

## 3. Functional requirements (restated precisely)

### 3.1 Registration (Stage 2)
- Operator can create a local profile: display name, consent flag, timestamp.
- Capture N face image samples across head angles and lighting conditions.
- Reject samples that fail quality checks (blur, multiple faces, no face, poor exposure).
- Generate and store face embeddings (vector, not raw biometric image, by default) and,
  optionally, body-appearance (re-ID) embeddings.
- Support profile review, embedding re-generation, and full deletion.

### 3.2 Detection & Tracking (Stage 1)
- Detect all people in each frame (bounding boxes, confidence).
- Assign a persistent tracker ID to each detected person across frames.
- Compute each track's center point and pixel-space error from frame center.
- Maintain FPS/latency telemetry and structured logs.

### 3.3 Identity verification (Stage 2/3)
- Given a set of tracked people, determine which (if any) match the **selected
  registered target** — not "who is the closest match among all registered users."
- Must be able to output **"no confident match"** rather than forcing a match.
- Use multiple observations over time (temporal consistency), not a single-frame
  decision, before committing to "this track = target."

### 3.4 Persistent tracking & gimbal command generation (Stage 1/3)
- Once a track is confirmed as the target, keep the tracker ID authoritative
  (do not re-run full identity verification every frame — that's what the tracker is for).
- Convert pixel-space target error into gimbal pan/tilt velocity commands with
  deadband, rate limiting, smoothing, and angle clamping.
- If gimbal saturates (target outside usable angular range), emit a high-level
  drone-movement **request** (not a raw motor command).

### 3.5 Loss & recovery (Stage 1/3)
- On loss/occlusion: snapshot last known state (bbox, center, velocity, direction,
  appearance embeddings, identity confidence, timestamp).
- Predict likely position using motion history (Kalman filter).
- Drive gimbal toward predicted direction; if not reacquired, execute a bounded
  search pattern.
- Re-verify candidates against stored identity+appearance data before resuming
  TRACKING — never auto-promote the nearest new detection.
- Timeout to a safe hover/search-stop state instead of searching indefinitely.

### 3.6 Safety gates (Stage 1 onward, enforced in software from day one)
Movement requests (gimbal or drone) must be suppressed whenever:
- Identity confidence < threshold
- Tracking confidence < threshold
- Target lost longer than configurable timeout
- Camera frame is stale (no new frame within expected interval)
- Perception process is in an error/crashed state
- (Later) flight-controller link is unavailable or drone is outside an approved state

## 4. Non-functional requirements

- Modular architecture: perception, identity, tracking, prediction, recovery,
  gimbal-command, drone-request, database, config, logging, viz, and testing are
  **separate modules/packages**, not one script.
- No cloud dependency for real-time detection/identification.
- All thresholds, paths, and tunables are configuration, not hard-coded.
- Type hints, docstrings (only where non-obvious), structured logging, pinned
  dependencies, unit + integration tests, linting, formatting, static typing.
- Biometric data (face/body embeddings) encrypted at rest; consent explicitly
  recorded per profile; raw images not stored by default.

## 5. Explicit non-goals

- Identifying or tracking anyone who has not explicitly registered and consented.
- Searching a scene for "any known person" — the system always operates against
  one **selected** target at a time.
- Fully autonomous flight decision-making — this project produces *requests*;
  a certified/tested flight controller and human-in-the-loop safety layer are
  out of scope until Stage 3+ and explicitly gated.
- Guaranteeing zero false accepts/rejects — the spec requires the system to
  **quantify and report** its error rates, not eliminate them.

## 6. Acceptance criteria categories (targets to be calibrated after Stage 1 baseline data — see `04_roadmap.md` §Testing)

Detection precision/recall, identity FAR/FRR, identity-switch count, track
continuity, recovery success rate & time, FPS, end-to-end latency, resource
usage (CPU/GPU/RAM/temp/power), max reliable range, and robustness under
motion blur, partial occlusion, similar-looking bystanders. Numeric targets
are deferred until we have a measured Stage 1 baseline — committing to numbers
before any data exists would be guessing, not engineering.
