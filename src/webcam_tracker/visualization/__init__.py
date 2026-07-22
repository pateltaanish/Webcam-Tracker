"""Debug visualization.

Draws bounding boxes, per-detection/per-track colors, target-selection
overlay, a simulated-gimbal widget, and a perf overlay onto frames for
interactive debugging. Read-only with respect to pipeline state (only
mutates the image it's given to draw onto).
"""

from webcam_tracker.visualization.colors import PALETTE, assign_colors, color_for_track_id
from webcam_tracker.visualization.draw import (
    draw_detections,
    draw_gimbal_widget,
    draw_perf_overlay,
    draw_recovery_overlay,
    draw_state_banner,
    draw_target_overlay,
    draw_tracked_people,
)

__all__ = [
    "PALETTE",
    "assign_colors",
    "color_for_track_id",
    "draw_detections",
    "draw_gimbal_widget",
    "draw_perf_overlay",
    "draw_recovery_overlay",
    "draw_state_banner",
    "draw_target_overlay",
    "draw_tracked_people",
]
