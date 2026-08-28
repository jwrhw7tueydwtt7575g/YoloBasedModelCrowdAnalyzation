"""Structured JSON logging setup.

Production entrypoints (``main.py``, ``src/mctracker/__main__.py``) call
``configure_logging()`` instead of ``logging.basicConfig`` so log records are
emitted as JSON with the standard fields (``asctime``, ``levelname``,
``name``, ``message``) plus any structured ``extra={...}`` keys the caller
attaches. If ``python-json-logger`` is not installed we fall back to plain
text — still better than nothing, and avoids a hard import on the
lightweight install.

The shape of the JSON record is fixed: ``time``, ``level``, ``logger``,
``message``, plus any ``extra`` keys. Tests assert on these names.
"""

from __future__ import annotations

import logging
from typing import Any


_FALLBACK_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


class _JsonFormatter(logging.Formatter):
    """Tiny pure-stdlib JSON formatter. Used only if python-json-logger
    is not installed. Output keys match python-json-logger's defaults so
    log shippers don't have to handle two formats."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Forward any structured extras (stream_id, stage, event, ...).
        for key, value in record.__dict__.items():
            if key in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "taskName",
            ):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: if the root logger already has our handler, leave it alone.
    """
    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, "_mctracker_json", False):
            return
    handler = logging.StreamHandler()
    handler._mctracker_json = True  # type: ignore[attr-defined]
    try:
        from pythonjsonlogger import jsonlogger  # type: ignore[import]

        handler.setFormatter(jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "time", "levelname": "level", "name": "logger"},
            json_ensure_ascii=False,
        ))
    except Exception:
        handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Convenience: log a structured event.

        log_event(log, logging.INFO, "stream reconnected",
                  stream_id="cam0", attempt=3)
    """
    logger.log(level, message, extra=fields)