"""Target-loss recovery.

On loss, snapshots last-known target state, predicts direction from motion
history, drives a bounded search pattern, and re-verifies candidates against
stored identity/appearance data before resuming tracking (see
docs/00_engineering_spec.md sec 3.5). Implemented starting Stage 1.8.
"""
