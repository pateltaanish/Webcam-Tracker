"""Unit tests for webcam_tracker.logging_utils.logger."""

from __future__ import annotations

import json
import logging

from webcam_tracker.logging_utils.logger import _JsonFormatter


def test_json_formatter_produces_valid_json_with_expected_fields() -> None:
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="webcam_tracker.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )

    formatted = formatter.format(record)
    payload = json.loads(formatted)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "webcam_tracker.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_json_formatter_includes_extra_fields() -> None:
    formatter = _JsonFormatter()
    logger = logging.getLogger("webcam_tracker.test.extra")
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        1,
        "track update",
        (),
        None,
        extra={"track_id": 42, "confidence": 0.87},
    )

    payload = json.loads(formatter.format(record))

    assert payload["extra"] == {"track_id": 42, "confidence": 0.87}
