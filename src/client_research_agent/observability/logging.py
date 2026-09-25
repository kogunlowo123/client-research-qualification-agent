"""Structured logging with correlation context and secret/PII scrubbing.

Every log line carries ``run_id``/``company``/``step`` from contextvars so a
single brief can be followed across agents, threads and Databricks job tasks.
A redaction processor removes credential-shaped values before emission.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog

_context: ContextVar[dict[str, Any] | None] = ContextVar("cra_log_context", default=None)

_SECRET_PATTERNS = (
    re.compile(r"dapi[0-9a-f]{32}(-\d)?", re.IGNORECASE),
    re.compile(r"(?:github_pat|ghp|gho|ghs)_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
_SENSITIVE_KEYS = frozenset({"token", "password", "secret", "authorization", "api_key", "apikey"})
_REDACTED = "[REDACTED]"


def scrub(value: Any) -> Any:
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(_REDACTED, value)
        return value
    if isinstance(value, dict):
        return {k: (_REDACTED if str(k).lower() in _SENSITIVE_KEYS else scrub(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(scrub(v) for v in value)
    return value


def _redact_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    return {k: (_REDACTED if k.lower() in _SENSITIVE_KEYS else scrub(v)) for k, v in event_dict.items()}


def _context_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in (_context.get() or {}).items():
        event_dict.setdefault(key, value)
    return event_dict


def configure_logging(level: str = "INFO", *, json_logs: bool = True, service_name: str = "cra") -> None:
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level {level!r}")
    logging.basicConfig(stream=sys.stdout, level=numeric, format="%(message)s", force=True)
    renderer: Any = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _context_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
    bind_context(service=service_name)


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


def bind_context(**values: Any) -> None:
    _context.set({**(_context.get() or {}), **values})


def current_context() -> dict[str, Any]:
    return dict(_context.get() or {})


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    token = _context.set({**(_context.get() or {}), **values})
    try:
        yield
    finally:
        _context.reset(token)
