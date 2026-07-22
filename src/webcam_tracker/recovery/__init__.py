"""Target-loss recovery (Stage 1.8).

On loss of the selected target, predicts where it went (from
motion_prediction's velocity estimate), drives a simulated search sweep, and
re-locks onto the nearest newly-appearing track near the predicted region.

Stage 1's reacquisition is IDENTITY-FREE -- a labeled placeholder for the
real face/re-id-gated reacquisition in Stage 2, which replaces the "nearest
new track" decision without changing this module's interface. See
controller.py for the full caveat.
"""

from webcam_tracker.recovery.controller import (
    RecoveryController,
    RecoveryState,
    RecoveryStatus,
)
from webcam_tracker.recovery.factory import create_recovery_controller

__all__ = [
    "RecoveryController",
    "RecoveryState",
    "RecoveryStatus",
    "create_recovery_controller",
]
