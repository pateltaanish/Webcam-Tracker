"""Structured (JSON) logging setup, shared by every module in the pipeline."""

from webcam_tracker.logging_utils.logger import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
