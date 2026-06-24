"""Structured JSON logging for Loki.

Replaces the CLI's plain ``%(name)s: %(message)s`` with one JSON object per line,
carrying correlation fields (``run_id``, ``session_id``, ``stage``, ``movie_slug``,
``lang``) bound per-task via a contextvar. Promtail/Alloy tails container stdout and
ships to Loki; every line is then filterable by ``run_id``.
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone

# Correlation context bound by the API request / Celery task and read by the formatter.
_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("log_ctx", default={})

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
}


def bind(**fields: object) -> contextvars.Token:
    """Bind correlation fields for the current context (returns a reset token)."""
    merged = {**_ctx.get(), **{k: v for k, v in fields.items() if v is not None}}
    return _ctx.set(merged)


def reset(token: contextvars.Token) -> None:
    _ctx.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_ctx.get())
        # promote any structured extras attached to the record
        for k, v in record.__dict__.items():
            if k not in _RESERVED and k not in payload and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(level: int = logging.INFO) -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
