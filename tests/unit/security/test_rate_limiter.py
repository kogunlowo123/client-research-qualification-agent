from __future__ import annotations

import threading

import pytest

from client_research_agent.security.rate_limiter import (
    BudgetExceededError,
    PrincipalRateLimiter,
    RateLimitExceededError,
    RunBudget,
    TokenBucket,
)
from client_research_agent.services.ports import LLMUsage
from client_research_agent.utils.errors import SecurityViolationError


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_token_bucket_refill() -> None:
    clock = Clock()
    bucket = TokenBucket(2, 1.0, clock=clock)
    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()
    assert bucket.retry_after() == pytest.approx(1.0)
    clock.now = 0.5
    assert bucket.available == pytest.approx(0.5)
    clock.now = 10.0
    assert bucket.available == pytest.approx(2.0)
    assert bucket.retry_after() == 0.0
    assert not bucket.try_acquire(3)
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(0, 1)


def test_principal_limiter_isolated_and_evicts() -> None:
    clock = Clock()
    limiter = PrincipalRateLimiter(1, 0.5, max_principals=2, clock=clock)
    assert limiter.allow("a")
    assert not limiter.allow("a")
    limiter.acquire("b")
    with pytest.raises(RateLimitExceededError) as info:
        limiter.acquire("b")
    assert info.value.retry_after_seconds == pytest.approx(2.0)
    assert isinstance(info.value, SecurityViolationError)
    limiter.allow("a")
    limiter.allow("c")
    assert limiter.tracked_principals == 2
    with pytest.raises(RateLimitExceededError) as too_big:
        limiter.acquire("d", cost=5)
    assert too_big.value.retry_after_seconds == float("inf")


def test_run_budget_limits() -> None:
    budget = RunBudget(max_tokens=100, max_cost=1.0, max_llm_calls=3)
    snapshot = budget.charge(LLMUsage(30, 20), cost=0.25)
    assert snapshot.tokens == 50
    assert snapshot.llm_calls == 1
    assert budget.remaining_tokens == 50
    budget.reserve(40)
    with pytest.raises(BudgetExceededError, match="tokens"):
        budget.reserve(60)
    with pytest.raises(BudgetExceededError, match="cost"):
        budget.reserve(1, estimated_cost=0.8)
    budget.charge(LLMUsage(1, 1))
    budget.charge(LLMUsage(1, 1))
    with pytest.raises(BudgetExceededError) as info:
        budget.charge(LLMUsage(1, 1))
    assert info.value.dimension == "llm_calls"
    assert budget.snapshot().llm_calls == 4
    with pytest.raises(ValueError, match="positive"):
        RunBudget(max_tokens=0)


def test_run_budget_thread_safe() -> None:
    budget = RunBudget(max_tokens=10_000_000, max_cost=1e9, max_llm_calls=10_000)

    def work() -> None:
        for _ in range(100):
            budget.charge(LLMUsage(5, 5), cost=0.01)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert budget.snapshot().tokens == 8000
    assert budget.snapshot().cost == pytest.approx(8.0)
