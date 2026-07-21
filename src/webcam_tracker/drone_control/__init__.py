"""Drone movement requests.

Converts gimbal-saturation/loss conditions into high-level movement
*requests* (e.g. "target left of usable range", "search requested") — never
raw motor commands. All requests pass through the safety gates described in
docs/02_architecture.md sec 5. Implemented starting Stage 1.7 (simulated),
real integration deferred to Stage 3+ with explicit sign-off.
"""
