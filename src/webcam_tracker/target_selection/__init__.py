"""Target selection.

Selects which track (Stage 1: manual selection by clicking a track) is the
current target the rest of the pipeline should follow. Stage 2+ replaces
manual selection with DB-driven identity selection behind the same interface.
"""

from webcam_tracker.target_selection.selector import TargetSelector, TargetStatus

__all__ = ["TargetSelector", "TargetStatus"]
