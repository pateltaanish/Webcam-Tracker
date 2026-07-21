"""Tracking state machine.

The authoritative coordinator: every other module feeds it observations, and
it alone drives gimbal/drone control outputs, gated by confidence and safety
conditions (see docs/02_architecture.md sec 4-5). Implemented in Stage 1.9.
"""
