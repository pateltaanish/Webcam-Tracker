# Staged Roadmap

This expands your Development Stages into concrete, checkpointed work. Each
checkpoint ends with something runnable/testable — we don't move to the next
one until the current one works and is verified together.

## Stage 1 — Desktop tracking prototype (no identity yet)

Goal: prove detection + tracking + gimbal-error math + loss/recovery
*mechanics* work, using only generic "any person" tracking (identity comes in
Stage 2). This deliberately defers the hardest part (identity) until the
scaffolding around it is solid.

1.1. Repo scaffold, virtual environment, dependency pinning, config system,
     structured logging skeleton. *(next step after this doc, pending your go-ahead)*
1.2. **DONE.** `video_input`: webcam + video file + folder-of-videos sources
     behind one interface (`FrameSource`). Webcam source uses `"auto"`
     device-index discovery (probes indices, platform-appropriate OpenCV
     backend ordering -- DSHOW before MSMF on Windows) so the same config
     works unmodified on a desktop with a USB webcam and a laptop with a
     built-in webcam, without hardcoding a device index. Verified against a
     real USB webcam on the desktop dev machine (`scripts\smoke_test_video_input.py`,
     negotiated 1280x720@30fps as configured). Not yet verified on a laptop
     with a built-in webcam. `scripts\list_cameras.py` enumerates all working
     indices on any machine if `"auto"` ever needs overriding.
1.3. **DONE.** `detection`: YOLO11n person-only inference wrapper
     (`PersonDetector`), filtered to whichever class the model itself calls
     "person" rather than an assumed class id. Weights auto-download to
     `models/yolo11n.pt` (gitignored) on first use -- see
     docs/model_licenses.md for the exact pinned file and its AGPL-3.0
     license. torch/torchvision installed from PyTorch's cu126 CUDA wheel
     index (see requirements-ml.txt) since plain `pip install torch` on
     Windows defaults to CPU-only; verified `torch.cuda.is_available()` on
     the RTX 3060 dev machine. Integration-tested against Ultralytics'
     bundled sample images (real inference, no network/test-data
     dependency). Two live-verification scripts:
     `scripts\smoke_test_detection.py` (headless, saves an annotated frame
     to disk) and `scripts\preview_detection.py` (live GUI window with
     boxes + FPS, detection only -- no persistent IDs until 1.4 tracking).
1.4. **DONE.** `tracking`: ByteTrack wrapper (`PersonTracker`, via the
     `trackers` package -- `supervision.ByteTrack` is deprecated, see
     docs/model_licenses.md) assigning persistent track IDs to detections.
     A new track is withheld until `tracking.minimum_consecutive_frames`
     consecutive matches confirm it (filters out one-frame flicker false
     positives becoming "tracked people"); "track started" events are
     logged. True identity-switch *detection/measurement* (was this new ID
     actually the same physical person?) is deferred to the Stage 1.10 test
     harness with labeled ground-truth video, not attempted inside the live
     tracker itself. Verified against real webcam frames
     (`scripts\smoke_test_tracking.py`): a single person's track ID stayed
     stable across 89/90 observed frames (99%), zero switches. Live GUI
     verification: `scripts\preview_tracking.py` colors each box by
     track_id (stable per-person color, unlike detection-stage's
     position-based coloring) -- try it with two people crossing paths to
     see where ByteTrack's motion-only matching can still swap IDs (that's
     the known limitation Re-ID in Stage 2 exists to reduce).
1.5. **DONE.** `visualization` + `perf_monitor`: on-screen boxes/IDs/FPS/latency
     overlay, formalized out of the ad-hoc drawing/FPS code that had been
     duplicated across all three preview scripts since Stage 1.2. `PerfMonitor`
     tracks rolling FPS and per-stage latency (`measure("detection")`,
     `measure("tracking")`, ...) with an injectable clock for deterministic
     unit tests, plus rate-limited structured logging
     (`perf_monitor.log_interval_seconds`) so continuous per-frame monitoring
     doesn't spam the log file. `draw_detections`/`draw_tracked_people`/
     `draw_perf_overlay` replace the inline `cv2.rectangle`/`cv2.putText`
     calls that used to be copy-pasted in each script. All three preview
     scripts refactored onto the new modules; re-verified end-to-end against
     the real webcam (`scripts\smoke_test_detection.py`).
1.6. **DONE.** `target_selection` (manual, Stage 1 stub): `TargetSelector`
     holds a selected track_id and reports per-frame `TargetStatus`
     (visible/not, target center, pixel error vs. frame center, normalized
     error) -- no identity verification yet, this is purely
     "which track_id is the target and where is it relative to center,"
     exercising the downstream pipeline end to end ahead of Stage 1.7's
     gimbal math. `select_at_point()` maps a click to whichever tracked
     box contains it (smallest box wins on overlap). Live verification:
     `scripts\preview_target_selection.py` (left-click to select,
     right-click or 'c' to clear, highlighted box + crosshair + error
     readout). Verified end-to-end against real webcam frames
     (`scripts\smoke_test_target_selection.py`, auto-clicking the first
     confirmed track): target stayed visible with zero "TARGET LOST" events
     across 29 frames while pixel error tracked real left/right movement
     smoothly (-193px to +28px), normalized error staying in the expected
     -1..1 range.
1.7. **DONE.** `gimbal_control` (simulated): `AxisController` (single-axis
     PID + deadband + rate limiting + angle clamp + anti-windup) used twice
     by `GimbalController` (pan + tilt), consuming target_selection's
     normalized error. No real motors -- see the module's own docstring for
     why building this now, without hardware, is still valuable (design/test
     the control logic against real tracking data; a real driver slots in
     later behind the same GimbalCommand interface). Includes an
     emergency-stop/resume state. Deliberately "dumb": it only reacts to the
     error it's given, no search/neutral-position policy of its own -- that
     belongs to state_machine (Stage 1.9). 29 unit tests, each PID term (P/I/D)
     verified against hand-derived expected numbers via a fake clock, 100%
     coverage on both controller files. Live verification:
     `scripts\preview_gimbal_control.py` (attitude-indicator-style widget,
     e-stop/resume bound to keys). Verified end-to-end against real webcam
     frames (`scripts\smoke_test_gimbal_control.py`): velocity ramped
     smoothly frame-to-frame (rate limiting working, no jumps), both axes
     correctly saturated at their configured angle limits (±170° pan, +60°
     tilt) while still reporting the commanded velocity being blocked --
     the signal drone_control will eventually consume.
     drone_control itself is deferred (not part of this step): saturation
     -> movement-request policy is more naturally driven by state_machine
     (Stage 1.9), so building it now would mean guessing at that interface
     rather than deriving it from a real state machine.
1.8. **DONE.** `motion_prediction` + `recovery` (generic, no identity yet).
     `motion_prediction`: a from-scratch constant-velocity Kalman filter
     (`KalmanFilter2D`, state = [x, y, vx, vy] in pixels) per track, managed
     by `MotionPredictor` (one filter per track_id, injectable clock for
     dt, coasts a just-lost track for `max_coast_seconds` so its last-known
     velocity is still readable). `recovery`: `RecoveryController`, a small
     loss->search->reacquire state machine (`RecoveryState`:
     IDLE/TRACKING/SEARCHING/REACQUIRED/GAVE_UP) that, on target loss,
     extrapolates the last-known position+velocity (capped at
     `prediction_horizon_seconds`), drives a simulated horizontal search
     sweep (a normalized error fed to the gimbal, biased toward the predicted
     direction), and re-locks onto the nearest *newly-appearing* track within
     `reacquire_radius_fraction` of the predicted spot -- **identity-free**,
     an explicitly-labeled placeholder for Stage 2's face/re-id-gated
     reacquisition that slots in at that exact decision point. It
     *recommends* a track_id rather than mutating the selection itself
     (selection stays owned by TargetSelector), and does not own overall
     system state (that's state_machine, 1.9). New `draw_recovery_overlay`
     (state banner + predicted-position marker + re-lock radius circle).
     24 new unit tests: Kalman velocity convergence on constant-velocity
     input, stationary-point stability, pure-lookahead non-mutation,
     coast-then-prune, and every recovery transition + the exact
     predicted-position extrapolation math, all on a fake clock (145 tests
     total, all passing; ruff + mypy src scripts clean). Live verification:
     `scripts\preview_recovery.py` (walk out of frame -> search -> walk back
     in -> auto re-lock, note the track-ID change it papers over).
     `scripts\smoke_test_recovery.py` verifies the loss path headlessly by
     *simulating* a loss (hiding the target from recovery after a tracking
     phase) and watching SEARCHING -> predicted extrapolation -> GAVE_UP.
     NOTE: live smoke-test run pending a person in frame at run time; the
     mechanics are covered exactly by the unit tests regardless.
1.9. **DONE.** `state_machine`: `TrackingStateMachine` is the authoritative
     coordinator. Perception (detector + tracker) runs upstream and feeds it
     tracked people each frame; it owns the four *decision* components
     (TargetSelector, MotionPredictor, RecoveryController, GimbalController)
     and collapses them into ONE `SystemStatus` per frame (state + target
     status + recovery status + gimbal command + `needs_drone_assist` +
     `reacquired_track_id`), logging every transition as a structured event.
     States are the Stage-1 (identity-free) subset of docs/02_architecture.md
     sec 4: IDLE, TRACKING, TEMPORARILY_OCCLUDED, RECOVERY_SEARCH,
     SAFE_HOVER_REQUESTED, STOPPED. A key behavior the state machine adds on
     top of the raw modules: a brief-dropout grace period
     (`occlusion_timeout_seconds`, ~= the tracker's lost_track_buffer) during
     which a missing target is TEMPORARILY_OCCLUDED -- the gimbal HOLDS and
     waits for the same track id to return -- before escalating to an active
     RECOVERY_SEARCH; this avoids swinging the camera (or grabbing a different
     person) on a few-frame detection gap. Reacquisition stays identity-free;
     the doc's REGISTERING/CANDIDATE_DETECTED/VERIFYING_IDENTITY states are
     the Stage-2 slot-in point (identity verification inserts at the reacquire
     decision), and `needs_drone_assist` (gimbal saturated while it should be
     following) is the concrete signal a future drone_control will consume.
     11 new unit tests wire the real components onto one fake clock and drive
     whole scenarios frame-by-frame (track -> occlude -> search -> give up, and
     -> reacquire, plus e-stop and drone-assist), 145 total. New
     `draw_state_banner`. Live verification: `scripts\smoke_test_state_machine.py`
     ran end-to-end against the real webcam -- IDLE -> TRACKING, then a
     simulated loss produced TEMPORARILY_OCCLUDED (frame 0) ->
     RECOVERY_SEARCH (frame 6, ~1s = occlusion timeout) ->
     SAFE_HOVER_REQUESTED (frame 32, ~5s later = recovery give-up), timing
     matching config exactly. `scripts\preview_state_machine.py` is the live
     GUI (state banner + the per-frame logic collapsed to one update() call).
     drone_control is still deferred but now has a defined interface to read
     (`SystemStatus.state` + `needs_drone_assist`).
1.10. **DONE — Stage 1 exit criterion met.** Two-part integration pass:
      (a) Automated: `tests/integration/test_pipeline.py` drives the WHOLE
      pipeline (real detector + tracker + state machine and all its real
      sub-components) over short "videos" built from Ultralytics' bundled
      sample images (real inference on real people, frame contents we
      control), with an injected fake clock so the time-based transitions are
      deterministic. Four scenarios, all passing: a single person is tracked
      and followed with a stable id; multiple people (bus.jpg, 4 detected) are
      all tracked while the state machine follows exactly one; a
      leave-then-return escalates TRACKING -> TEMPORARILY_OCCLUDED ->
      RECOVERY_SEARCH and recovers; and — the key one — a person who leaves
      long enough to come back as a **new track id** (>lost_track_buffer) is
      reacquired by recovery and followed again, which is exactly the
      "comes back as a new ID" case, now handled end to end.
      (b) Qualitative tool: `scripts/integration_report.py` runs the pipeline
      over a real video file/folder (or the webcam) and prints a structured
      report — state distribution, timed transition list, and loss/search/
      reacquire/give-up tallies — for eyeballing recorded clips (one person,
      two crossing, leave/re-enter). Verified live against the webcam: over a
      2s clip the target held TRACKING 70% of frames and correctly absorbed
      10 single-frame detection dropouts as brief TEMPORARILY_OCCLUDED blips
      that each returned to TRACKING WITHOUT escalating to a search — the
      occlusion grace period doing its job. Also added optional `clock`
      injection to the gimbal/motion/recovery/state_machine factories to make
      this deterministic (fake clock) and video-time (report tool) driving
      possible.
      145 total (134 unit + 11 state-machine unit) + 4 new full-pipeline
      integration tests, ruff + mypy clean.

**Stage 1 is complete.** The full desktop tracking prototype — detection,
tracking, target selection, simulated gimbal control, motion prediction,
target-loss recovery, and the authoritative state machine — is built, unit-
and integration-tested, and verified against real webcam data. Everything is
identity-FREE by design; the "which specific person is this" problem is
Stage 2. `drone_control` remains deferred with a defined interface to read
(`SystemStatus.state` + `needs_drone_assist`).

## Stage 2 — Registration & identity database

**Design doc written: `docs/database_design.md`** (read it first). Confirmed
decisions (2026-07-22): encryption keyed by an **operator passphrase**
(Argon2id → AES-256-GCM envelope encryption, key never stored); face model is
**InsightFace SCRFD + ArcFace**; build order is a **face-first vertical slice**
(database → registration → face → identity fusion, then add body Re-ID).

2.1. **DONE.** `database`: local encrypted store (see
     `docs/database_design.md`) for profiles + embeddings + consent metadata.
     `ProfileStore` = SQLite + application-layer AES-256-GCM on sensitive
     columns (display name, embedding vectors), keyed by an operator passphrase
     via Argon2id → wrapped data key (envelope encryption; passphrase never
     stored). Passphrase unlock **fails closed** (wrong passphrase → the GCM
     auth check rejects it, no partial unlock); `add/get/list/revoke/delete`
     people + embeddings; consent events + an audit log; `change_passphrase`
     re-wraps the key without re-encrypting data. Argon2id cost tuned to a
     ~0.5s unlock (time_cost=4, 512 MiB — measured). New deps
     (`cryptography`, `argon2-cffi`) pinned in `requirements-identity.txt`.
     24 unit tests including the security-critical ones: wrong-passphrase-
     fails-closed, plaintext-name/embedding-bytes-absent-from-the-file
     (encrypted at rest), delete-really-removes-data, passphrase rotation.
     Store lives at `data/identity/` (gitignored). No face model yet (2.3).
2.2. Registration CLI/flow: capture samples, quality checks (blur/exposure/
     single-face), multi-angle prompts, embedding generation (face via
     SCRFD+ArcFace, optional Re-ID via OSNet), storage.
2.3. `face_recognition` + `reid` modules: embedding extraction + cosine-similarity
     comparison against stored profiles, with confidence thresholds and an
     explicit "unknown" outcome.
2.4. `identity` fusion module: combines face-match, reid-match, and temporal
     consistency into the confidence score consumed by the state machine.
     Replace Stage 1's placeholder reacquisition logic with real
     identity-gated reacquisition.
2.5. Test harness: known-target vs. unregistered-person vs. wrong-registered-user
     scenarios, measuring FAR/FRR on your own captured data.

## Stage 3 — Embedded migration

3.1. Finalize onboard computer purchase (per `03_onboard_computer.md`,
     pending your R1 decision).
3.2. Model conversion/quantization (TensorRT or HailoRT depending on choice),
     benchmarked against Stage 1's desktop numbers.
3.3. Camera driver integration, hardware-abstraction layer for camera/gimbal/FC.
3.4. Startup service, watchdog, crash recovery, thermal/power monitoring,
     structured telemetry, remote debugging, safe shutdown.
3.5. On-hardware benchmark pass against the acceptance-criteria categories in
     `00_engineering_spec.md` §6, now with real numbers.

## Gimbal & drone-control preparation (parallel track, starts in Stage 1)

Simulated gimbal controller + visualization + unit tests + safety clamps +
e-stop state are built in Stage 1 (§1.7) precisely so the *interface* is
proven before Stage 3 swaps in a real driver. Drone-control stays a
request-only interface behind the safety gates in `02_architecture.md` §5
through every stage — no raw motor access is ever wired up as part of this
project's software; that integration explicitly requires your own
hardware-in-the-loop testing and sign-off when the time comes.

## What happens next

If this spec/architecture/hardware direction looks right, the next concrete
step is **Stage 1.1**: scaffold the repository, set up the Python virtual
environment, and pin initial dependencies (PyTorch w/ CUDA for your 3090,
Ultralytics, OpenCV, etc.) — then implement §1.2 (video input) as the first
piece of runnable code, tested against your webcam before anything else is
built on top of it.

I have **not** written any code yet, on purpose — confirm this direction
(or tell me what to change) and I'll start Stage 1.1.
