# Risks & Unrealistic Assumptions

Read this before we pick hardware. Several requirements are in tension with the
250 g / $350 constraints, and I'd rather surface that now than discover it after
you've bought parts.

## R1 — The 250 g limit and the "run everything onboard" goal are in direct tension (HIGH severity)

A modern AI companion computer capable of running a detector + face pipeline +
Re-ID pipeline at usable speed (Jetson Orin Nano class) weighs on the order of
**120–150 g bare** (module + carrier board, no cooling, no camera, no gimbal,
no battery, no frame, no motors). A rough weight budget for the *rest* of the
airframe (3D-printed frame, 4 micro motors, props, AIO ESC/FC, LiPo battery,
receiver, wiring, a small 2-axis gimbal, a camera module) realistically lands
around **150–190 g** even when you optimize hard. Add those together and you
are already over 250 g before accounting for cooling (a passive Jetson
heatsink alone can be 20–40 g; without one it throttles under sustained load).

For comparison: consumer sub-250 g drones (e.g. DJI Mini class) achieve this
weight only with a custom ASIC/SoC purpose-built for the airframe — not an
off-the-shelf dev-kit SBC. We don't have that option at hobbyist scale.

**This means the 250 g target and "Jetson-class onboard AI" are not
simultaneously achievable with today's off-the-shelf parts**, at least not on
the first build.

**Recommended mitigation (this is a plan, not a decision — flag if you'd
rather go a different direction):**
1. Build and flight-test the full avionics/perception stack on a **slightly
   larger "validation" frame first** (roughly 400–600 g class — still tiny,
   still cheap to 3D print, but with margin). This lets us prove the software
   and hardware pipeline works before fighting for grams.
2. Once the pipeline is proven, do a dedicated **weight-optimization pass**:
   lighter SoM-only carrier boards, a lighter/quantized model set, a smaller
   battery (shorter flight time) — to approach 250 g as a final, focused
   engineering effort, not a day-1 constraint on every decision.
3. Alternative: relax to a lighter compute option from day one (Raspberry Pi 5
   + Hailo-8L, ~70 g total) and accept a smaller/slower model set. See
   `03_onboard_computer.md` for the full tradeoff — this is viable but with
   real accuracy/flexibility costs.

I'll ask you explicitly which path you want before we lock in hardware
purchases (§ "Decision needed" below).

## R2 — 30 FPS with the full identity+Re-ID pipeline on embedded hardware is optimistic (MEDIUM)

30 FPS end-to-end (detect + track + face-verify + Re-ID + gimbal command) on a
desktop RTX 3090 is easy. On any realistic ≤150 g onboard computer it is not
guaranteed, especially if face recognition and Re-ID run every frame. The
standard technique — and what we'll build in from Stage 1 — is to run the
*detector + tracker* every frame (cheap) and run the *heavier* face/Re-ID
verification only periodically (e.g. every 5–10 frames, or only during
VERIFYING_IDENTITY/RECOVERY states) while the lightweight tracker carries
continuity in between. Treat 30 FPS as the tracker/gimbal-loop rate, and
identity re-confirmation as a slower background rate. We'll measure real
numbers on your actual onboard hardware in Stage 3 rather than promise a
figure now.

## R3 — Face recognition will not work "when the face is not visible" — that's what Re-ID is for (MEDIUM, expectation-setting)

Face embeddings require a visible, reasonably front-on face. When the target's
back is turned, face-based identity confirmation is unavailable by design —
not a bug to fix. This is precisely why the spec calls for a **separate**
person Re-ID (body/clothing appearance) model: it degrades gracefully but is
itself not perfectly reliable (clothing changes fool it, similar-looking
bystanders in similar clothing fool it). The system's honest worst case is:
"low confidence, ask for re-verification via face when it becomes visible
again." We will never claim identity certainty from Re-ID alone.

## R4 — Model licensing is not just a formality (MEDIUM)

Several strong, popular detection models (Ultralytics YOLOv8/YOLO11) are
licensed **AGPL-3.0**, which has real implications if this project is ever
distributed or run as a network service in anything other than an
AGPL-compatible open-source form (Ultralytics sells a commercial license as
the alternative). We'll pick primary models accordingly and document exact
license terms per model file in `docs/model_licenses.md` once we pin actual
weight files in Stage 1 — see `02_architecture.md` §Licensing.

## R5 — Identity systems have irreducible error rates (LOW severity, but must be stated up front per your own requirements)

No combination of face recognition + Re-ID achieves zero false-accept/false-reject
in the real world, especially at the small model sizes embedded hardware
requires. The system will be built to **quantify** FAR/FRR on your own test
data (Stage 2 testing harness) and to **fail toward "unknown"/no-action**
rather than fail toward "confidently wrong." This is a design principle, not
a solved problem — flag it now so it isn't a surprise later.

## R6 — Outdoor + indoor without a low-light requirement simplifies camera choice, but wind/vibration on a <250 g airframe will stress the gimbal and tracker (LOW-MEDIUM)

Small, light airframes are more wind-sensitive and vibration-prone per gram of
thrust than larger drones. Motion blur and frame jitter will be a real testing
scenario (already in your spec's test list) — we are not deferring it, just
noting it compounds with R1/R2.

## Decision needed before hardware purchase

Given R1, do you want to:
- **(A)** Build/test the full pipeline on a larger (~500 g) validation frame
  first, then do a dedicated weight-optimization pass toward 250 g later, or
- **(B)** Commit to the lighter/cheaper compute (Raspberry Pi 5 + Hailo-8L)
  from day one to stay closer to 250 g immediately, accepting a smaller,
  less flexible model set and a harder embedded-deployment path?

I lean toward (A) — it de-risks the software (the hard, novel part of this
project) before fighting grams (a hardware-selection problem with diminishing
returns on debugging time) — but this is your call and your budget. We can
revisit after Stage 1 gives us real performance numbers.
