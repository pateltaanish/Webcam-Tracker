"""Structured logging.

Every module gets its logger via ``get_logger(__name__)``. Output is JSON
(one object per line) so that later stages (Stage 3 telemetry, or just
grepping/parsing logs during Stage 1/2 testing) can consume it programmatically
instead of scraping free-text log lines. Call ``configure_logging`` once, at
process startup, before any module logs anything.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_CONFIGURED = False

# Attributes already present on every LogRecord — anything else passed via
# `logger.info(..., extra={...})` is "extra" application data we want to
# surface in the JSON output.
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO", log_dir: Path | None = None, json_format: bool = True
) -> None:
    """Configure the root logger once for the whole process.

    Args:
        level: Standard logging level name (DEBUG/INFO/WARNING/ERROR).
        log_dir: If given, logs are also written to ``<log_dir>/webcam_tracker.log``
            in addition to stderr. Created if it doesn't exist.
        json_format: Use structured JSON output; if False, use a plain
            human-readable formatter (handy while reading logs live in a
            terminal during early development).
    """
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level)

    formatter: logging.Formatter
    if json_format:
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "webcam_tracker.log", encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)

    root.handlers = handlers
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Safe to call before ``configure_logging``
    (Python's logging module queues output on the root handler once configured)."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
