# Onboard Computer Selection

Read `01_risks_and_assumptions.md` §R1 first — the 250 g target and a
Jetson-class computer are in tension, and this comparison is written with
that tradeoff explicit rather than hidden.

## Comparison

| Criterion | **Jetson Orin Nano Super Dev Kit** (primary) | **Raspberry Pi 5 (8GB) + Hailo-8L AI HAT+** (lower-cost/lighter) | **Jetson Orin NX 16GB module** (higher-performance, for reference) |
|---|---|---|---|
| Board weight (bare, no cooling/camera/gimbal) | ~130–140 g (full carrier + module) | ~46 g (Pi5) + ~25 g (HAT) ≈ **~71 g total** | ~35–45 g module only, but needs a carrier board (+50–80 g) — comparable total to Orin Nano devkit |
| Cooling required | Active (fan) recommended under sustained load; adds ~10–20 g + power draw | Passive heatsink sufficient for our workload | Active cooling required, more thermal headroom needed than Orin Nano |
| Power consumption | 7–25 W configurable power modes | ~5–12 W (Pi5 + Hailo combined) | 10–25 W+ |
| AI acceleration | Ampere GPU, 67 TOPS INT8 (Super mode) | Hailo-8L NPU, 13 TOPS INT8 | Ampere GPU, up to 100 TOPS INT8 |
| Compute for our pipeline (detector+tracker+face+reid) | Comfortable margin, room to run multiple models per frame | Workable for detector+tracker every frame; face/reid must run at reduced rate to hold FPS | Most margin of the three |
| Camera compatibility | MIPI-CSI (2x), USB, broad support | MIPI-CSI (2x on Pi5), USB | MIPI-CSI, USB |
| GPIO / UART / I2C / SPI / USB / PWM | Full GPIO header, UART/I2C/SPI, USB 3.2, PWM via GPIO or companion MCU | Full GPIO header, UART/I2C/SPI, USB 3.0, PWM via GPIO | Depends on chosen carrier board; generally full support |
| Model-conversion / deployment path | **TensorRT** — direct PyTorch→ONNX→TensorRT, best-documented path for exactly our model set (YOLO, ONNX-exportable face/reid models) | ONNX→**HailoRT compiler** — works, but a narrower toolchain, smaller community, more friction for a first embedded project | Same as Orin Nano (TensorRT) |
| Software ecosystem maturity | Excellent — full Ubuntu/L4T, CUDA, PyTorch, huge community, most tutorials assume Jetson | Good general Pi ecosystem, but Hailo-specific ML tooling is newer/thinner | Same as Orin Nano |
| Boot time | ~15–25 s typical | ~10–15 s typical | Similar to Orin Nano |
| Reliability / availability | Widely available, NVIDIA-backed, long track record in robotics/drones | Widely available, Raspberry Pi Foundation-backed | Available but pricier, more niche for hobbyist drones |
| Cost | **$249** (Super Dev Kit) | **~$80 (Pi5 8GB) + ~$70 (Hailo-8L HAT+) ≈ $150** | **~$599 module alone** (before any carrier) |
| Flight-controller integration | UART/USB to a standard FC (e.g. running Betaflight/ArduPilot/PX4) — well-trodden path in the drone/robotics community | Same UART/USB path, equally viable | Same |
| Gimbal-controller integration | UART/I2C/PWM — standard | Same | Same |
| Fits our $350 compute+camera+gimbal budget? | Tight — leaves ~$100 for camera+gimbal | Comfortable — leaves ~$200 for camera+gimbal | **No** — blows the entire budget on compute alone |
| Fits path toward 250 g AUW? | Very difficult without R1's mitigation plan | Realistic — best option if 250 g is a day-1 hard constraint | Same difficulty as Orin Nano, worse |

### A fourth option worth knowing about: Luxonis OAK-D-Lite

Not in the main table because it's a different category — a **camera with an
onboard AI accelerator (Myriad X VPU) built in**, not a general-purpose
computer. It can run the detector (and potentially a second model) directly
on the camera module itself, offloading work from whatever host computer you
pair it with. Weight is roughly 60–90 g *including the camera*, and it uses
the OpenVINO toolchain (mature, but distinct from both TensorRT and
HailoRT — a third ecosystem to learn). Worth a serious look during Stage 3
hardware finalization if the Pi5+Hailo path turns out too slow for detector+
tracker at once, since it effectively adds a second, camera-integrated
accelerator to the system. I'm not recommending it as primary now because
it complicates the architecture (two accelerators/toolchains instead of one)
before we've even validated the software — but it's a good escape hatch.

## Recommendation

**Primary: Jetson Orin Nano Super Dev Kit ($249).** Best software ecosystem
for a CV beginner (TensorRT/PyTorch/CUDA, the vast majority of embedded-CV
tutorials and community troubleshooting target Jetson), most compute margin
for running detector+tracker+face+reid without aggressive frame-skipping,
proven track record in hobbyist/research drones. Tradeoff, per R1: this
consumes ~70% of the $350 budget and makes 250 g AUW unrealistic on the first
build — I'm recommending we validate on a larger (~500 g) test frame first,
then do a dedicated weight pass (e.g., migrating to a bare Orin Nano module
on a minimal third-party carrier, which can shave 30–50 g).

**Lower-cost/lighter alternative: Raspberry Pi 5 (8GB) + Hailo-8L AI HAT+
(~$150 total).** Nearly half the cost, roughly half the weight of the Jetson
devkit. This is the pragmatic choice if you'd rather stay closer to 250 g
from day one (Path B in R1) and accept a narrower, less-documented model
toolchain and reduced margin for running all four models at full rate.

**Higher-performance alternative (reference only, not fitting this
project's constraints): Jetson Orin NX 16GB.** Roughly 1.5–2x the compute of
Orin Nano, but ~$599 for the module alone (before a carrier board) — blows
the entire hardware budget on compute alone, with no meaningful weight
advantage. I would not recommend this unless the budget or weight target
changes substantially (e.g., a second, larger sibling drone down the line).

## Decision

Deferred to you per R1's "Decision needed" — I've defaulted the roadmap to
**Path A (Jetson Orin Nano, validate on a larger test frame first)** since it
best serves the software-correctness goals of this project, but this is
easily reversible: nothing in Stage 1 (desktop prototype) depends on this
choice, and Stage 3 planning won't finalize hardware purchases until we've
seen real Stage 1 performance numbers on your RTX 3090 to extrapolate from.
