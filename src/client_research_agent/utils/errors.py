"""Typed error hierarchy.

Errors are split by *retryability* so resilience policies can make decisions
without string matching: ``TransientError`` subclasses are retried and count
against circuit breakers; everything else fails fast.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(AgentError):
    """Invalid or missing configuration. Never retried."""


class TransientError(AgentError):
    """A failure that may succeed on retry (timeouts, 429, 5xx)."""


class UpstreamTimeoutError(TransientError):
    pass


class RateLimitedError(TransientError):
    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class UpstreamServiceError(TransientError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CircuitOpenError(AgentError):
    """Raised without calling the dependency because its breaker is open."""

    def __init__(self, name: str, retry_in_seconds: float) -> None:
        super().__init__(f"circuit '{name}' is open; retry in {retry_in_seconds:.1f}s")
        self.name = name
        self.retry_in_seconds = retry_in_seconds


class CrawlPolicyViolationError(AgentError):
    """URL disallowed by robots.txt, the domain allow-list, or SSRF protection."""


class SecurityViolationError(AgentError):
    """Prompt injection, poisoned content or policy breach detected."""


class OutputValidationError(AgentError):
    """LLM output could not be parsed into the required schema after repair attempts."""


class InsufficientEvidenceError(AgentError):
    """Not enough grounded evidence to proceed with a step."""
