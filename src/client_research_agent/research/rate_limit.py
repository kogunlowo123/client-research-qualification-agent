"""Thread-safe per-host token-bucket rate limiter.

Each host gets its own bucket refilled at ``rate`` tokens/second up to
``burst`` tokens. ``acquire`` *reserves* a token under the lock (the balance
may go negative) and sleeps outside it, so concurrent callers for the same
host are serialised fairly without holding the lock while waiting, and
callers for different hosts never block each other.

``set_min_interval`` lets robots.txt ``Crawl-delay`` slow a host down; it can
only ever make a host *slower* than the configured default.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from client_research_agent.research.url_guard import normalize_host


@dataclass
class _Bucket:
    rate: float
    capacity: float
    tokens: float
    updated: float


class HostRateLimiter:
    def __init__(
        self,
        rate_per_second: float,
        *,
        burst: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        if burst < 1.0:
            raise ValueError("burst must be >= 1")
        self._default_rate = rate_per_second
        self._burst = burst
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def _bucket(self, host: str, now: float) -> _Bucket:
        bucket = self._buckets.get(host)
        if bucket is None:
            bucket = _Bucket(self._default_rate, self._burst, self._burst, now)
            self._buckets[host] = bucket
        return bucket

    def set_min_interval(self, host: str, seconds: float) -> None:
        """Ensure at least ``seconds`` between requests to ``host`` (never speeds a host up)."""
        if seconds <= 0:
            return
        host = normalize_host(host)
        with self._lock:
            bucket = self._bucket(host, self._clock())
            bucket.rate = min(bucket.rate, 1.0 / seconds)
            bucket.capacity = 1.0
            bucket.tokens = min(bucket.tokens, bucket.capacity)

    def rate_for(self, host: str) -> float:
        with self._lock:
            bucket = self._buckets.get(normalize_host(host))
            return bucket.rate if bucket else self._default_rate

    def reserve(self, host: str) -> float:
        """Take one token for ``host`` and return how long the caller must wait before using it."""
        host = normalize_host(host)
        with self._lock:
            now = self._clock()
            bucket = self._bucket(host, now)
            elapsed = max(0.0, now - bucket.updated)
            bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.rate)
            bucket.updated = now
            bucket.tokens -= 1.0
            return 0.0 if bucket.tokens >= 0 else -bucket.tokens / bucket.rate

    def acquire(self, host: str) -> float:
        """Block until a request to ``host`` is permitted; returns the seconds waited."""
        wait = self.reserve(host)
        if wait > 0:
            self._sleep(wait)
        return wait
