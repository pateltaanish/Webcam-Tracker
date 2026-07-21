"""Gimbal command generation.

Converts pixel-space target error into pan/tilt velocity commands (PID +
deadband + rate limiting + angle clamping + e-stop). Stage 1 output is
simulated/logged only, never wired to real motors. Implemented in Stage 1.7.
"""
