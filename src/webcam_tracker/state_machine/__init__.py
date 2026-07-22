"""Tracking state machine (Stage 1.9).

The authoritative coordinator: perception (detection + tracking) feeds it
tracked people each frame, and it drives one system state + one coordinated
set of control outputs (target status, recovery status, gimbal command),
logging every transition as a structured event. It owns the selection /
prediction / recovery / gimbal decision components; a future drone_control
reads its `needs_drone_assist` signal. See state_machine.py for the state set
and how it maps onto docs/02_architecture.md sec 4.
"""

from webcam_tracker.state_machine.factory import create_state_machine
from webcam_tracker.state_machine.state_machine import (
    SystemStatus,
    TrackingState,
    TrackingStateMachine,
)

__all__ = [
    "SystemStatus",
    "TrackingState",
    "TrackingStateMachine",
    "create_state_machine",
]
