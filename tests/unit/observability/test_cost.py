from __future__ import annotations

import threading

import pytest

from client_research_agent.observability.cost import (
    DEFAULT_PRICING,
    EndpointPricing,
    PricingTable,
    TokenCostTracker,
)
from client_research_agent.observability.metrics import Metrics
from client_research_agent.services.ports import LLMUsage


def test_pricing_math() -> None:
    pricing = EndpointPricing(input_dbu_per_million=10.0, output_dbu_per_million=20.0)
    assert pricing.dbus(LLMUsage(1_000_000, 500_000)) == pytest.approx(20.0)
    table = PricingTable(prices={"a": pricing}, usd_per_dbu=0.5)
    assert table.for_endpoint("a") is pricing
    assert table.for_endpoint("unknown") is table.fallback
    assert "databricks-claude-sonnet-4" in PricingTable.default().prices
    assert DEFAULT_PRICING["databricks-gte-large-en"].output_dbu_per_million == 0.0


def test_tracker_totals_and_groups() -> None:
    metrics = Metrics()
    table = PricingTable(prices={"chat": EndpointPricing(1_000_000, 2_000_000)}, usd_per_dbu=0.1)
    tracker = TokenCostTracker(table, metrics=metrics)
    entry = tracker.record("qualify", "chat", LLMUsage(10, 5))
    assert entry.dbus == pytest.approx(20.0)
    assert entry.usd == pytest.approx(2.0)
    tracker.record("brief write", "chat", LLMUsage(1, 1))
    totals = tracker.totals()
    assert (totals.calls, totals.prompt_tokens, totals.completion_tokens, totals.total_tokens) == (
        2,
        11,
        6,
        17,
    )
    assert totals.usd == pytest.approx(2.3)
    assert set(tracker.by_step()) == {"qualify", "brief write"}
    assert tracker.by_endpoint()["chat"].calls == 2
    flat = tracker.as_metrics()
    assert flat["llm_total_tokens"] == 17.0
    assert flat["llm_tokens.brief_write"] == 2.0
    assert tracker.estimate("chat", LLMUsage(1, 0)) == pytest.approx(0.1)
    assert metrics.counter("llm.tokens.prompt", endpoint="chat", step="qualify") == 10
    assert metrics.counter("llm.cost.usd", endpoint="chat") == pytest.approx(2.3)
    tracker.reset()
    assert tracker.entries() == []


def test_negative_usage_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenCostTracker().record("s", "e", LLMUsage(-1, 0))


def test_thread_safe_accumulation() -> None:
    tracker = TokenCostTracker()

    def work() -> None:
        for _ in range(250):
            tracker.record("s", "databricks-claude-sonnet-4", LLMUsage(2, 2))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.totals().total_tokens == 4000
