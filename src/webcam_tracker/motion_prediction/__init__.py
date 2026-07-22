"""Motion prediction (Stage 1.8).

Per-track constant-velocity Kalman filter producing a smoothed position and
an inferred velocity (px/s) for each tracked person. The velocity estimate is
what lets the recovery module guess which direction a just-lost target went;
the smoothed position is also a cleaner signal than the raw jittery box
center. Pure logic, no camera/OpenCV -- see kalman.py for the filter itself.
"""

from webcam_tracker.motion_prediction.factory import create_motion_predictor
from webcam_tracker.motion_prediction.kalman import KalmanFilter2D
from webcam_tracker.motion_prediction.predictor import MotionPredictor, TrackMotion

__all__ = [
    "KalmanFilter2D",
    "MotionPredictor",
    "TrackMotion",
    "create_motion_predictor",
]
