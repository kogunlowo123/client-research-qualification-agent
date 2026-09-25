"""Translate Databricks SDK / REST failures into the package's typed error hierarchy.

The Databricks SDK, the Vector Search (AI Search) client and ``requests`` each
raise their own exception types. Resilience policies only understand
``TransientError`` subclasses, so every adapter funnels foreign exceptions
through :func:`map_databricks_error` before they reach ``call_with_retry``.
Classification uses class names and ``status_code`` attributes so that it does
not need to import every optional SDK.
"""

from __future__ import annotations

from client_research_agent.utils.errors import (
    AgentError,
    ConfigurationError,
    RateLimitedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)


class DatabricksRequestError(AgentError):
    """A non-retryable Databricks request failure (bad request, missing resource, failed SQL)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_RATE_LIMITED = frozenset({"TooManyRequests", "ResourceExhausted", "RequestLimitExceeded"})
_TIMEOUT = frozenset({"DeadlineExceeded", "Timeout", "ReadTimeout", "ConnectTimeout"})
_UNAVAILABLE = frozenset(
    {"InternalError", "TemporarilyUnavailable", "Unknown", "DataLoss", "ConnectionError", "Aborted"}
)
_AUTH = frozenset({"Unauthenticated", "PermissionDenied"})


def _class_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retry_after(exc: BaseException) -> float | None:
    value = getattr(exc, "retry_after_secs", None)
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    return None


def map_databricks_error(exc: BaseException, context: str) -> AgentError:
    """Return the typed equivalent of ``exc``; ``AgentError`` instances pass through unchanged."""
    if isinstance(exc, AgentError):
        return exc
    names = _class_names(exc)
    status = _status_code(exc)
    message = f"{context}: {type(exc).__name__}: {exc}"
    if names & _RATE_LIMITED or status == 429:
        return RateLimitedError(message, retry_after_seconds=_retry_after(exc))
    if names & _TIMEOUT or isinstance(exc, TimeoutError) or status in (408, 504):
        return UpstreamTimeoutError(message)
    if names & _UNAVAILABLE or isinstance(exc, ConnectionError) or (status is not None and status >= 500):
        return UpstreamServiceError(message, status_code=status)
    if names & _AUTH or status in (401, 403):
        return ConfigurationError(f"{message} (check the service principal's grants)")
    return DatabricksRequestError(message, status_code=status)
