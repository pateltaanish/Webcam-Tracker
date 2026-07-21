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
1.4. `tracking`: ByteTrack wrapper assigning persistent IDs. Verify: IDs stay
     stable across a video of one person walking around; log ID-switch events.
1.5. `visualization` + `perf_monitor`: on-screen boxes/IDs/FPS/latency overlay.
1.6. `target_selection` (manual, Stage 1 stub): click/key-select a track ID
     as "the target" — no identity verification yet, just to exercise the
     downstream pipeline end to end.
1.7. `gimbal_control` (simulated): pixel error → normalized error → simulated
     pan/tilt velocity, deadband, clamping. Output logged/displayed, no real
     motors. Unit tests for the math independent of any camera.
1.8. `motion_prediction` + `recovery` (generic, no identity yet): on track
     loss, predict direction from Kalman velocity, simulate a search sweep,
     re-lock on the nearest *newly appearing* track only if it appears near
     the predicted region (a placeholder for real identity-based
     reacquisition, clearly labeled as such).
1.9. `state_machine` wiring all of the above together, with structured event
     logging.
1.10. Integration test pass: run against a handful of recorded test videos
      (single person, two people crossing paths, person leaving/re-entering
      frame) and confirm logs/behavior make sense. This is our Stage 1 exit
      criterion — **not** perfect tracking, just correct mechanics we can
      build identity on top of.

## Stage 2 — Registration & identity database

2.1. `database`: local encrypted store (see `docs/database_design.md`, to be
     written alongside this stage) for profiles + embeddings + consent
     metadata.
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
