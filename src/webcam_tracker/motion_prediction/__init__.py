"""Motion prediction.

Per-track Kalman filter (constant-velocity model) producing predicted
position/direction, used both for smooth tracking and to drive the recovery
controller's search direction after target loss. Implemented in Stage 1.8.
"""
