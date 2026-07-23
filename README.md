# Webcam-Tracker

Consent-based person-identification and tracking system, being built toward
a <=250 g drone. **Stage 1 (current): desktop tracking prototype.**

This project only identifies people who have knowingly registered and
consented. It is not designed for surveillance or identifying strangers. See
`docs/00_engineering_spec.md` for the full spec and non-goals.

## Documentation

- [`docs/00_engineering_spec.md`](docs/00_engineering_spec.md) -- requirements, confirmed constraints, acceptance-criteria categories
- [`docs/01_risks_and_assumptions.md`](docs/01_risks_and_assumptions.md) -- known tensions (weight vs. compute budget, etc.) -- read this before making hardware decisions
- [`docs/02_architecture.md`](docs/02_architecture.md) -- pipeline design, model comparisons + selections, state machine
- [`docs/03_onboard_computer.md`](docs/03_onboard_computer.md) -- embedded hardware comparison + recommendation
- [`docs/04_roadmap.md`](docs/04_roadmap.md) -- staged build plan
- [`docs/database_design.md`](docs/database_design.md) -- Stage 2 identity + encrypted-database design (consent, encryption, schema) -- read before Stage 2 code

## Setup (Windows, Stage 1)

Run these from a terminal in the repo root (`e:\Webcam-Tracker`).

1. **Create the virtual environment** (only needed once):
   ```
   py -3.12 -m venv .venv
   ```
   What it does: creates an isolated Python environment in `.venv/` so this
   project's dependencies never conflict with anything else on your machine.
   Success looks like: a new `.venv/` folder appears, silently, no output.
   Common error: `py: command not found` -- install Python 3.12+ from
   python.org and make sure "Add py launcher to PATH" was checked during
   install.

2. **Install dependencies**:
   ```
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-ml.txt
   .venv\Scripts\python.exe -m pip install -e . --no-deps
   ```
   `requirements-ml.txt` installs torch/torchvision/ultralytics (detection)
   and trackers/supervision (tracking) -- it's a much heavier download
   (~2-3 GB) and pulls torch from PyTorch's own CUDA wheel index rather than
   plain PyPI (see the comments at the top of that file for why, and what to
   do on a CPU-only machine).
   What it does: installs pinned runtime + dev dependencies, then installs
   this project itself in "editable" mode (`-e .`) so `import webcam_tracker`
   works from anywhere without reinstalling after every code change.
   `--no-deps` skips re-resolving dependencies we already pinned above.
   Success looks like: `Successfully installed webcam-tracker-0.1.0` as the
   last line.
   Common error: a `UnicodeDecodeError` mentioning this README -- means the
   README got saved in a non-UTF-8 encoding; re-save it as UTF-8.

3. **Copy the local environment file** (for future secrets/overrides -- not
   needed yet in Stage 1, but the pattern is in place):
   ```
   copy .env.example .env
   ```

4. **Run the test suite** to confirm everything works:
   ```
   .venv\Scripts\python.exe -m pytest
   ```
   Success looks like: `N passed` in green/plain text, with a coverage table
   above it. If a test fails, don't build on top of it -- fix it first.

5. **Lint, format-check, and type-check** (run these before committing):
   ```
   .venv\Scripts\python.exe -m ruff check .
   .venv\Scripts\python.exe -m ruff format --check .
   .venv\Scripts\python.exe -m mypy src scripts
   ```
   `ruff format --check` reports files that *would* be reformatted without
   changing them; drop `--check` to actually apply formatting.

## Project layout

```
src/webcam_tracker/   Application code, one package per pipeline stage
  config/             Typed config loading (YAML defaults + env overrides)
  logging_utils/      Structured JSON logging
  video_input/        Camera/file/folder frame sources          (Stage 1.2)
  detection/          Person detector wrapper                   (Stage 1.3)
  tracking/           Multi-object tracker (ByteTrack)           (Stage 1.4)
  visualization/      Debug overlay drawing                      (Stage 1.5)
  perf_monitor/       FPS/latency/resource tracking               (Stage 1.5)
  target_selection/   Manual -> DB-driven target selection        (Stage 1.6)
  gimbal_control/     Pixel error -> simulated gimbal commands    (Stage 1.7)
  drone_control/      High-level movement requests + safety gates (Stage 1.7)
  motion_prediction/  Kalman-filter motion estimate                (Stage 1.8)
  recovery/           Target-loss search/reacquisition             (Stage 1.8)
  state_machine/      Authoritative tracking state machine         (Stage 1.9)
  face_recognition/   Face detect + embed + match                  (Stage 2)
  reid/               Body-appearance re-identification             (Stage 2)
  identity/           Fuses face+reid+temporal into confidence      (Stage 2)
  database/           Encrypted registered-user store               (Stage 2)
configs/default.yaml  All non-secret tunables
tests/unit/           Unit tests, one module per package
tests/integration/    End-to-end pipeline tests
docs/                 Specs, architecture, roadmap (see above)
```

## Configuration

Every tunable lives in `configs/default.yaml`. Override any value without
editing that file by setting an environment variable
`WEBCAM_TRACKER_<SECTION>__<FIELD>` (e.g. `WEBCAM_TRACKER_VIDEO__SOURCE=1`),
either in your shell or in a local `.env` file (gitignored -- copy
`.env.example` to get started). See comments in `configs/default.yaml` and
`src/webcam_tracker/config/settings.py` for details.

## Video input (Stage 1.2)

`video.source: "auto"` (the default) probes for the first working camera and
uses it -- this is what lets the same config work unmodified on a desktop
with a USB webcam and a laptop with a built-in webcam, since it never
assumes a fixed device index.

Two helper scripts:

- `scripts\list_cameras.py` -- enumerates every working camera index on the
  current machine (headless, prints a table). Run this first on a new
  machine, or if `"auto"` picks the wrong camera on a machine with more than
  one.
- `scripts\preview_webcam.py` -- opens a live window showing the configured
  source with an FPS overlay, for visually confirming capture works. Press
  `q` or Esc to close.
- `scripts\smoke_test_video_input.py` -- headless (no GUI window) check that
  reads a handful of real frames and prints their shape/fps/timestamps; used
  to verify video_input end-to-end without a display.

## Person detection (Stage 1.3)

`src/webcam_tracker/detection` wraps a YOLO11n model (Ultralytics),
filtered to the "person" class only. Weights auto-download to
`models/yolo11n.pt` on first use (gitignored -- not committed; see
`docs/model_licenses.md` for the license). Verification scripts:

- `scripts\smoke_test_detection.py` -- headless: grabs one real frame from
  your configured camera, runs detection, prints results, and saves an
  annotated image to `data\debug_output\detection_smoke.jpg` for review.
- `scripts\preview_detection.py` -- live GUI window with detection boxes +
  confidence + FPS drawn on the real-time feed. Press `q` or Esc to close.
  This is detection only, not tracking -- no persistent ID across frames yet
  (that's Stage 1.4).

## Person tracking (Stage 1.4)

`src/webcam_tracker/tracking` wraps ByteTrack (via the `trackers` package)
to assign a persistent `track_id` to each detected person across frames --
so the pipeline can say "this is the same person as last frame," not just
"there's a person here." A new person takes a couple of frames to first
appear (tracks aren't reported until confirmed by
`tracking.minimum_consecutive_frames` consecutive matches, filtering out
one-frame flicker). Verification scripts:

- `scripts\smoke_test_tracking.py` -- headless: runs detection + tracking on
  ~3 seconds of real frames and reports how stable the track ID(s) stayed.
- `scripts\preview_tracking.py` -- live GUI window, box color keyed by
  `track_id` (same person = same color across frames, unlike detection's
  position-based coloring). Try it with two people crossing paths to see
  where motion-only tracking can still swap IDs -- that's a real limitation,
  not a bug, and part of what Stage 2's Re-ID work will reduce.

## Visualization & performance monitoring (Stage 1.5)

`src/webcam_tracker/perf_monitor` (`PerfMonitor`) tracks rolling FPS and
per-stage latency (e.g. how long detection vs. tracking took within a
frame), and rate-limits structured perf-snapshot log lines so continuous
monitoring doesn't spam the log file. `src/webcam_tracker/visualization`
(`draw_detections`, `draw_tracked_people`, `draw_perf_overlay`) draws boxes/
IDs/FPS onto a frame. All three preview scripts (`preview_webcam.py`,
`preview_detection.py`, `preview_tracking.py`) now share these instead of
each duplicating their own FPS-counter and box-drawing code.

## Manual target selection (Stage 1.6)

`src/webcam_tracker/target_selection` (`TargetSelector`) lets you pick which
tracked person is "the target" and reports, every frame, whether they're
still visible and their pixel/normalized error relative to frame center --
the raw signal Stage 1.7's simulated gimbal controller will consume. This is
manual selection only (click a person); Stage 2+ swaps in DB-driven identity
selection behind the same interface, so nothing downstream has to change.

- `scripts\preview_target_selection.py` -- live GUI window. **Left-click** a
  tracked person to select them as the target (right-click or press 'c' to
  clear). The target gets a fixed-color highlight box, a crosshair at both
  frame-center and target-center connected by a line, and a pixel-error /
  normalized-error readout. If the target's track is lost, you'll see a red
  "TARGET LOST" warning instead of a stale box.
- `scripts\smoke_test_target_selection.py` -- headless: auto-"clicks" the
  first confirmed track and reports its status/error over a few seconds, to
  verify the whole chain without needing to click anything.

## Simulated gimbal control (Stage 1.7)

`src/webcam_tracker/gimbal_control` turns target_selection's normalized
error into simulated pan/tilt commands via a PID controller per axis, with
deadband, rate limiting, angle clamping, and anti-windup -- **no physical
gimbal is required or driven**; there isn't one in this project yet. The
point of building this now is to design and test the control logic against
real tracking data, so a real motor driver can slot in later (Stage 3)
behind the same interface without touching this code or anything upstream
of it. It also includes an emergency-stop/resume state.

- `scripts\preview_gimbal_control.py` -- live GUI window: same click-to-select
  as target selection, plus an attitude-indicator-style widget (top-right)
  showing the simulated gimbal's current position -- green normally, red
  when an axis is saturated (hit its limit), orange when e-stopped. Press
  `e` to emergency-stop, `r` to resume.
- `scripts\smoke_test_gimbal_control.py` -- headless: prints pan/tilt
  velocity + angle every frame as the auto-selected target moves.

PID gains in `configs/default.yaml` are explicitly untuned placeholders --
there's no physical gimbal yet to tune against; see the comments there.

## Motion prediction & target-loss recovery (Stage 1.8)

Two modules that handle a target *leaving frame* instead of just dropping it:

- `src/webcam_tracker/motion_prediction` (`MotionPredictor`) runs a
  from-scratch constant-velocity Kalman filter per track, smoothing the
  jittery box center and, more importantly, *inferring each track's velocity*
  (the detector only ever reports position). That velocity is what lets us
  guess which way a lost target was heading.
- `src/webcam_tracker/recovery` (`RecoveryController`) is a small
  loss -> search -> reacquire state machine. When the selected target
  disappears, it extrapolates where it went, drives a simulated search sweep
  (fed to the gimbal), and re-locks onto the nearest *newly-appearing* track
  near the predicted spot.

**Important — this reacquisition is identity-FREE.** It grabs whoever walks
into the predicted region, not necessarily the original person. It's a
deliberately-labeled placeholder for Stage 2's real face/re-id-gated
reacquisition, which replaces just that decision without changing the
interface. It does *not* solve the "person leaves and comes back with a new
track ID" problem on its own -- that needs identity (Stage 2).

- `scripts\preview_recovery.py` -- live GUI: click-to-select, then walk out
  of frame and watch the RECOVERY banner + predicted-position marker + re-lock
  radius circle, with the gimbal widget sweeping in search; walk back in to
  see it auto re-lock. Also keeps the gimbal e-stop ('e') / resume ('r') keys.
- `scripts\smoke_test_recovery.py` -- headless: tracks you for a bit, then
  *simulates* a loss (hides the target from recovery) and prints the search
  playing out to GAVE_UP.

## Tracking state machine (Stage 1.9)

`src/webcam_tracker/state_machine` (`TrackingStateMachine`) is the "brain"
that ties Stage 1 together. Perception (detection + tracking) runs upstream
and hands it tracked people each frame; it coordinates target selection,
motion prediction, recovery, and the gimbal into ONE authoritative state and
one set of outputs per frame (a `SystemStatus`), and logs every state change
as a structured event. The states (identity-free Stage 1 subset of
`docs/02_architecture.md` §4):

- **IDLE** -- nothing selected
- **TRACKING** -- following the selected target
- **TEMPORARILY_OCCLUDED** -- target briefly missing; hold position and wait
  for the same track id to return (avoids over-reacting to a few-frame gap)
- **RECOVERY_SEARCH** -- missing too long; run the recovery search sweep
- **SAFE_HOVER_REQUESTED** -- search gave up
- **STOPPED** -- emergency stop

`SystemStatus.needs_drone_assist` (set when the gimbal is maxed out while it
should be following) is the concrete signal a future `drone_control` will
read to decide when to move the drone body. Reacquisition is still
identity-free; Stage 2's identity check inserts at that exact decision point.

- `scripts\preview_state_machine.py` -- live GUI with the state banner
  (top-left, colored by state). This is the full pipeline; note how the
  per-frame logic is now a single `state_machine.update(...)` call.
- `scripts\smoke_test_state_machine.py` -- headless: follows you, then
  simulates a loss and prints the state progression through to safe-hover.

## Integration pass (Stage 1.10)

The Stage 1 exit check, in two parts:

- `tests\integration\test_pipeline.py` -- runs the **whole** pipeline (real
  detector + tracker + state machine) over short "videos" built from bundled
  sample images, asserting the mechanics of the three roadmap scenarios:
  single person tracked, multiple people handled, and a person who leaves and
  returns as a **new track ID** getting reacquired (the exact "comes back as
  a new ID" case). Runs as part of `pytest`.
- `scripts\integration_report.py` -- point it at a recorded clip (or a folder
  of clips, or the webcam) and it prints a structured report: state
  distribution, a timed list of state transitions, and loss / search /
  reacquisition / give-up counts. Use it to eyeball real footage of the three
  scenarios:
  ```
  .venv\Scripts\python.exe scripts\integration_report.py data\clips\cross.mp4
  ```

## Status

**Stage 1 is complete** (repo scaffold 1.1 through the integration pass 1.10)
-- the full desktop tracking prototype is built, unit- and integration-tested,
and verified against real webcam data on the desktop USB webcam (GPU: RTX
3060, CUDA confirmed available): track IDs stable, target selection solid,
gimbal rate-limiting + saturation correct, the state machine steps
IDLE -> TRACKING -> (on loss) TEMPORARILY_OCCLUDED -> RECOVERY_SEARCH ->
SAFE_HOVER_REQUESTED with timing matching config, and the full pipeline
reacquires a person who returns as a new track ID. 149 tests (145 unit + 4
full-pipeline integration), ruff + mypy clean. Not yet verified against a
laptop's built-in webcam -- run `scripts\list_cameras.py` there first.
Everything through Stage 1 is identity-FREE by design (reacquisition re-locks
the nearest returning track, not a verified person).

**Stage 2 (registration & identity) is underway.**

- **2.1 (done):** the `database` module -- a local, encrypted
  (`data/identity/`, gitignored) profile store for registered people + their
  embeddings + consent metadata, keyed by an operator passphrase (Argon2id ->
  AES-256-GCM envelope encryption; passphrase never stored, unlock fails
  closed).
- **2.2 (done):** the `face_recognition` module (InsightFace SCRFD + ArcFace
  512-d embeddings) and the `registration` flow (quality gates + consent-first
  enrollment). Enroll a consenting person from the webcam with
  `.venv\Scripts\python.exe scripts\register_person.py` -- it stores face
  *embeddings* only, never images.

Needs the Stage 2 deps:
`.venv\Scripts\python.exe -m pip install -r requirements-identity.txt` (the
face model auto-downloads ~280 MB on first use). Next is face *matching*
against the store (2.3) then identity fusion into the state machine (2.4) --
see `docs/04_roadmap.md` and `docs/database_design.md`.
