"""Simulated gimbal controller: turns target pixel error into pan/tilt
commands. This never drives real hardware -- see the module docstring in
__init__.py for why we build this without a physical gimbal.
"""

from __future__ import annotations

from dataclasses import dataclass

from webcam_tracker.gimbal_control.axis_controller import AxisController
from webcam_tracker.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GimbalCommand:
    """One frame's simulated gimbal output."""

    pan_velocity_deg_s: float
    tilt_velocity_deg_s: float
    pan_angle_deg: float
    tilt_angle_deg: float
    pan_saturated: bool
    tilt_saturated: bool
    target_visible: bool
    emergency_stopped: bool


class GimbalController:
    """Combines a pan AxisController and a tilt AxisController.

    Scope note: this class only ever reacts to the error it's given (or
    "no target" -> hold). It does NOT decide search patterns, return-to-
    neutral behavior, or anything about *why* a target is or isn't visible
    -- that's the state_machine's job (Stage 1.9), which will call this
    differently depending on overall system state (e.g. feed it a sweep
    pattern's error during RECOVERY_SEARCH). Keeping this class "dumb" is
    deliberate: it's easier to test and reason about a pure error-to-command
    function than one that also encodes policy about when to search.
    """

    def __init__(self, pan: AxisController, tilt: AxisController) -> None:
        self._pan = pan
        self._tilt = tilt
        self._emergency_stopped = False

    def compute(
        self, normalized_pan_error: float | None, normalized_tilt_error: float | None
    ) -> GimbalCommand:
        """`None` for either error means "target not visible" -> hold position.

        Both errors must be given together or not at all (target_selection
        always produces both from one TargetStatus) -- pass
        `status.normalized_error[0]`/`[1]` if you have a TargetStatus, or
        `(None, None)` if `status` itself is None or `status.visible` is False.
        """
        target_visible = normalized_pan_error is not None and normalized_tilt_error is not None

        if self._emergency_stopped or not target_visible:
            pan_velocity, pan_angle, pan_saturated = self._pan.hold()
            tilt_velocity, tilt_angle, tilt_saturated = self._tilt.hold()
        else:
            assert normalized_pan_error is not None
            assert normalized_tilt_error is not None
            pan_velocity, pan_angle, pan_saturated = self._pan.update(normalized_pan_error)
            tilt_velocity, tilt_angle, tilt_saturated = self._tilt.update(normalized_tilt_error)

        return GimbalCommand(
            pan_velocity_deg_s=pan_velocity,
            tilt_velocity_deg_s=tilt_velocity,
            pan_angle_deg=pan_angle,
            tilt_angle_deg=tilt_angle,
            pan_saturated=pan_saturated,
            tilt_saturated=tilt_saturated,
            target_visible=target_visible,
            emergency_stopped=self._emergency_stopped,
        )

    def emergency_stop(self) -> None:
        """Immediately zero velocity and refuse further motion until resume()."""
        if not self._emergency_stopped:
            logger.warning("Gimbal emergency stop engaged")
        self._emergency_stopped = True
        self._pan.reset()
        self._tilt.reset()

    def resume(self) -> None:
        if self._emergency_stopped:
            logger.info("Gimbal emergency stop cleared")
        self._emergency_stopped = False

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped
