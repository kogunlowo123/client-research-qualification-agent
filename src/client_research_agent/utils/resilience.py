"""Retry and circuit-breaker primitives used around every remote dependency.

The breaker is a classic three-state machine (closed -> open -> half-open)
guarded by a lock so it is safe under the thread pools used for parallel
fetches and multi-query retrieval.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import ParamSpec, TypeVar

from client_research_agent.utils.errors import CircuitOpenError, RateLimitedError, TransientError

P = ParamSpec("P")
T = TypeVar("T")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._state = BreakerState.CLOSED

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if (
            self._state is BreakerState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self._reset_timeout
        ):
            self._state = BreakerState.HALF_OPEN

    def before_call(self) -> None:
        with self._lock:
            self._maybe_half_open()
            if self._state is BreakerState.OPEN and self._opened_at is not None:
                remaining = self._reset_timeout - (self._clock() - self._opened_at)
                raise CircuitOpenError(self.name, max(remaining, 0.0))

    def on_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._state = BreakerState.CLOSED

    def on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is BreakerState.HALF_OPEN or self._failures >= self._threshold:
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()

    def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        self.before_call()
        try:
            result = func(*args, **kwargs)
        except TransientError:
            self.on_failure()
            raise
        self.on_success()
        return result


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 20.0
    jitter: float = 0.25

    def backoff(self, attempt: int, error: BaseException | None = None) -> float:
        if isinstance(error, RateLimitedError) and error.retry_after_seconds is not None:
            return min(error.retry_after_seconds, self.max_backoff_seconds)
        base: float = min(self.initial_backoff_seconds * 2.0 ** (attempt - 1), self.max_backoff_seconds)
        spread = base * self.jitter
        return max(0.0, base + random.uniform(-spread, spread))  # noqa: S311  # nosec B311 - jitter, not crypto


def call_with_retry(
    func: Callable[[], T],
    *,
    policy: RetryPolicy,
    breaker: CircuitBreaker | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Invoke ``func`` retrying only ``TransientError``; a breaker short-circuits when open."""
    attempt = 0
    while True:
        attempt += 1
        try:
            if breaker is not None:
                return breaker.call(func)
            return func()
        except TransientError as exc:
            if attempt >= policy.max_attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            sleep(policy.backoff(attempt, exc))
