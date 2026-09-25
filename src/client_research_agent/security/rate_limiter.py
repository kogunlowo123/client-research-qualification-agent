"""Unbounded-consumption controls (OWASP LLM10).

* :class:`TokenBucket` / :class:`PrincipalRateLimiter` - per-principal request
  rate limiting for the serving API.
* :class:`RunBudget` - hard caps on LLM tokens, LLM calls and estimated cost for
  a single research run so a poisoned page or a reasoning loop cannot burn
  unbounded spend.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from client_research_agent.services.ports import LLMUsage
from client_research_agent.utils.errors import SecurityViolationError


class RateLimitExceededError(SecurityViolationError):
    def __init__(self, principal_id: str, retry_after_seconds: float) -> None:
        super().__init__(f"rate limit exceeded for '{principal_id}'; retry in {retry_after_seconds:.2f}s")
        self.principal_id = principal_id
        self.retry_after_seconds = retry_after_seconds


class BudgetExceededError(SecurityViolationError):
    def __init__(self, dimension: str, limit: float, attempted: float) -> None:
        super().__init__(f"run budget exceeded: {dimension} {attempted:g} > limit {limit:g}")
        self.dimension = dimension
        self.limit = limit
        self.attempted = attempted


class TokenBucket:
    def __init__(
        self, capacity: float, refill_per_second: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if capacity <= 0 or refill_per_second <= 0:
            raise ValueError("capacity and refill_per_second must be positive")
        self._capacity = float(capacity)
        self._rate = float(refill_per_second)
        self._clock = clock
        self._tokens = float(capacity)
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated = now

    def try_acquire(self, cost: float = 1.0) -> bool:
        if cost > self._capacity:
            return False
        with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    def retry_after(self, cost: float = 1.0) -> float:
        with self._lock:
            self._refill()
            deficit = cost - self._tokens
            return 0.0 if deficit <= 0 else deficit / self._rate

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


class PrincipalRateLimiter:
    """Independent token bucket per principal with LRU eviction to bound memory."""

    def __init__(
        self,
        capacity: float = 30.0,
        refill_per_second: float = 0.5,
        *,
        max_principals: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._rate = refill_per_second
        self._max = max_principals
        self._clock = clock
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    def _bucket(self, principal_id: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(principal_id)
            if bucket is None:
                bucket = TokenBucket(self._capacity, self._rate, clock=self._clock)
                self._buckets[principal_id] = bucket
                while len(self._buckets) > self._max:
                    self._buckets.popitem(last=False)
            else:
                self._buckets.move_to_end(principal_id)
            return bucket

    def allow(self, principal_id: str, cost: float = 1.0) -> bool:
        return self._bucket(principal_id).try_acquire(cost)

    def acquire(self, principal_id: str, cost: float = 1.0) -> None:
        bucket = self._bucket(principal_id)
        if not bucket.try_acquire(cost):
            wait = bucket.retry_after(cost) if cost <= self._capacity else float("inf")
            raise RateLimitExceededError(principal_id, wait)

    @property
    def tracked_principals(self) -> int:
        with self._lock:
            return len(self._buckets)


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    tokens: int
    llm_calls: int
    cost: float
    max_tokens: int
    max_llm_calls: int
    max_cost: float


class RunBudget:
    """Per-run hard limits. ``reserve`` checks before a call; ``charge`` records actual usage."""

    def __init__(
        self, *, max_tokens: int = 400_000, max_cost: float = 25.0, max_llm_calls: int = 200
    ) -> None:
        if max_tokens <= 0 or max_cost <= 0 or max_llm_calls <= 0:
            raise ValueError("budget limits must be positive")
        self._max_tokens = max_tokens
        self._max_cost = max_cost
        self._max_calls = max_llm_calls
        self._tokens = 0
        self._cost = 0.0
        self._calls = 0
        self._lock = threading.Lock()

    def reserve(self, estimated_tokens: int, estimated_cost: float = 0.0) -> None:
        with self._lock:
            self._check(self._tokens + estimated_tokens, self._cost + estimated_cost, self._calls + 1)

    def charge(self, usage: LLMUsage, cost: float = 0.0) -> BudgetSnapshot:
        with self._lock:
            self._tokens += usage.total_tokens
            self._cost += cost
            self._calls += 1
            self._check(self._tokens, self._cost, self._calls)
            return self._snapshot()

    def _check(self, tokens: int, cost: float, calls: int) -> None:
        if tokens > self._max_tokens:
            raise BudgetExceededError("tokens", self._max_tokens, tokens)
        if cost > self._max_cost:
            raise BudgetExceededError("cost", self._max_cost, round(cost, 6))
        if calls > self._max_calls:
            raise BudgetExceededError("llm_calls", self._max_calls, calls)

    def _snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            tokens=self._tokens,
            llm_calls=self._calls,
            cost=round(self._cost, 6),
            max_tokens=self._max_tokens,
            max_llm_calls=self._max_calls,
            max_cost=self._max_cost,
        )

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot()

    @property
    def remaining_tokens(self) -> int:
        with self._lock:
            return max(0, self._max_tokens - self._tokens)
