# Onboard Computer Selection

**Status: Compute + camera DECIDED (2026-07-31). Flight controller + frame
deliberately left open — see §2.** The original three-way comparison
(Jetson Orin Nano vs. Pi 5+Hailo-8L vs. Jetson Orin NX) that used to live in
this file is superseded by that decision; it's preserved in git history if
you want to revisit the reasoning, not duplicated here.

## 1. Decision: Raspberry Pi 5 (8GB) + Raspberry Pi AI Camera (IMX500)

No separate NPU accelerator (no Hailo-8L / AI HAT+). Division of labor:

| Pipeline stage | Runs where | Notes |
|---|---|---|
| Person detection (YOLO11n) | **On-sensor**, IMX500's built-in inference accelerator | Model goes through Sony's Model Compression Toolkit → `packerOut.zip` → `imx500-package` → `.rpk`, loaded onto the camera at runtime via `picamera2`'s `IMX500` device class. Ultralytics officially supports IMX500 export for YOLOv8n and YOLO11n specifically — this project's already-selected detector — so no model swap is needed for this stage. The camera returns bounding boxes as capture metadata alongside each frame; the Pi 5 CPU never runs the detector. |
| Tracking (ByteTrack) | Pi 5 CPU | Unchanged from desktop — not a neural net, cheap either way. Consumes the boxes read from camera metadata instead of a `PersonDetector.infer()` call. |
| Face detect + embed (SCRFD/ArcFace), Re-ID (OSNet) | Pi 5 CPU, via `onnxruntime` (CPU execution provider) | No accelerator backs this stage now — see the open risk below. |

**Weight/power estimate:** ~50–60 g for the two boards bare, more like
60–75 g once cabled/mounted — dramatically lighter than the Jetson-class
figures R1 was originally worried about, and lighter than the Pi5+Hailo
figure this doc previously carried, since there's no separate accelerator
board. Power: Pi 5 CPU load (tracking + periodic face/Re-ID) plus the
camera's on-sensor inference (on the order of 100 mW) is roughly **4–9 W**
combined — well under the Pi5+Hailo estimate, since the AI HAT+'s own 8–14 W
draw under load is gone entirely.

**Toolchain:** one conversion path (IMX500/Sony MCT, one-time, for the
detector only) plus plain `onnxruntime` CPU inference for everything else —
no HailoRT, no TensorRT. Simplest toolchain of any option considered so far.

## 2. Open risk carried by this decision: CPU-only face/Re-ID throughput

Dropping the Hailo NPU means face recognition and Re-ID have no accelerator
margin. Real Pi 5 CPU benchmarks for this exact model family:

| Face model combo | Pi 5 CPU time | Rate |
|---|---|---|
| `scrfd_10g` + `arcface_r50` — **this is what `buffalo_l` actually is, the pack currently pinned in `configs/default.yaml:209` and documented in `docs/database_design.md`/`docs/model_licenses.md`** | ~669 ms | ~1.5 FPS |
| `scrfd_2.5g` + `arcface_mobilefacenet` — the "lightweight backbone" `docs/02_architecture.md` §3.3 says was selected, but wasn't what got implemented | ~194 ms | ~5.2 FPS |

This is the model-pack inconsistency flagged at the top of this review: as
currently configured, the identity stage would run at ~1.5 FPS on this
hardware, which eats a large fraction of the state machine's
`VERIFYING_IDENTITY` (~2 s) and occlusion (~1–2 s) timeouts on its own,
before Re-ID or state-machine overhead. Swapping to the lightweight pack
(`buffalo_s`, or `scrfd_2.5g`+MobileFaceNet directly) is a one-line config
change and gets into a workable range for the periodic (not every-frame)
cadence R2 already calls for. **Action item before Stage 3.2:** make and
record that model-pack decision explicitly (it's a licensing/accuracy
tradeoff, not just a performance one — see `docs/model_licenses.md`), and
benchmark OSNet's real Pi-class CPU time too (only anecdotal, non-Pi-verified
numbers exist for it right now).

If CPU-only proves insufficient even with the lightweight pack, the Hailo
HAT+ isn't foreclosed — it can be added later to carry just the face/Re-ID
stage, since the detector stays on-sensor either way and doesn't compete for
the same accelerator.

## 3. Flight controller + frame: intentionally left open

No specific board or airframe is chosen. Because the only link between the
perception/compute stack and the airframe is a UART carrying MAVLink, FC and
frame selection is decoupled from everything in §1 — pick later, based on
availability and the weight/thrust budget once real Pi5+camera weight is
measured. Two hard constraints apply to whatever is picked, both non-negotiable
given how this project's `drone_control` module is designed (movement
*requests*, never raw motor access):

1. **Firmware must be ArduPilot (GUIDED mode) or PX4 (OFFBOARD mode).**
   Both accept externally-supplied position/velocity/attitude setpoints from
   a companion computer over MAVLink — this is exactly the shape of
   `drone_control`'s output. **Betaflight does not qualify** — it only
   speaks MAVLink for outbound telemetry, with no mechanism to accept
   autonomous setpoints from a companion computer. This rules out a large
   share of cheap FPV-racing boards, not just an edge case, so check firmware
   support before any FC purchase, not just UART presence.
2. **A free, correctly-configured UART on the FC**, 3.3 V TTL (matches the
   Pi 5's GPIO UART directly), both ends on the same baud rate (57600 or
   115200 are the common MAVLink defaults). Some smaller boards share their
   companion-computer-designated port (e.g. PX4's TELEM2) with GPS or a
   telemetry radio — confirm one is actually free on the specific board
   before buying.

Given those two constraints are met, frame/motor/battery selection is a pure
mechanical/electrical sizing question against the combined AUW (compute +
camera + FC + frame + battery), consistent with R1's existing plan: validate
on a larger frame first, do a dedicated weight pass once real numbers exist.

If the gimbal ends up FC-managed (MAVLink Gimbal Protocol v2) rather than
driven directly from the Pi 5 (PWM/I2C), that reinforces the same firmware
constraint — still ArduPilot or PX4 only.

## 4. What's still open

- Face-model-pack decision (§2) — blocks trusting the CPU-only identity
  stage's real-world timing.
- `video_input` now has a `picamera2`/`libcamera`-backed `FrameSource`
  (`PiCameraSource`, `raspi` branch) — set via `WEBCAM_TRACKER_VIDEO__SOURCE=picamera`
  in a local `.env`, per-machine as with every other source. This only covers
  **capture**: the detector still runs on the Pi 5 CPU through the normal
  `PersonDetector` path, same as desktop. Reading detection boxes off the
  camera's on-sensor metadata (so the Pi 5 CPU never runs the detector, per
  §1) is still open — it needs the IMX500 export (§2/`.rpk`) done first, plus
  an additive `detections` field on `VideoFrame` so boxes can arrive with the
  frame without an interface change for `detection`/`tracking` downstream.
- FC board + frame selection (§3), once weight/thrust numbers are real.
- On-hardware benchmark of the full pipeline (Stage 3.5) — nothing in this
  file is measured yet, all of it is spec-sheet/third-party-benchmark derived.
