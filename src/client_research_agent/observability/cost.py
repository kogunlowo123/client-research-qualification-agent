"""LLM token and cost accounting.

Databricks Foundation Model APIs bill pay-per-token workloads in DBUs per
million tokens. The default table below is an *estimate* for budgeting and
dashboards (override it from config with your contract's rates); totals are
reported both in DBUs and in USD at a configurable DBU price.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from client_research_agent.observability.metrics import Metrics
from client_research_agent.services.ports import LLMUsage


@dataclass(frozen=True, slots=True)
class EndpointPricing:
    input_dbu_per_million: float
    output_dbu_per_million: float

    def dbus(self, usage: LLMUsage) -> float:
        return (
            usage.prompt_tokens * self.input_dbu_per_million
            + usage.completion_tokens * self.output_dbu_per_million
        ) / 1_000_000


DEFAULT_PRICING: Mapping[str, EndpointPricing] = {
    "databricks-claude-sonnet-4": EndpointPricing(42.857, 214.286),
    "databricks-meta-llama-3-3-70b-instruct": EndpointPricing(7.143, 21.429),
    "databricks-gte-large-en": EndpointPricing(1.857, 0.0),
    "databricks-bge-large-en": EndpointPricing(1.429, 0.0),
}
DEFAULT_FALLBACK_PRICING = EndpointPricing(42.857, 214.286)


@dataclass(frozen=True, slots=True)
class PricingTable:
    prices: Mapping[str, EndpointPricing]
    usd_per_dbu: float = 0.07
    fallback: EndpointPricing = DEFAULT_FALLBACK_PRICING

    def for_endpoint(self, endpoint: str) -> EndpointPricing:
        return self.prices.get(endpoint, self.fallback)

    @classmethod
    def default(cls) -> PricingTable:
        return cls(prices=dict(DEFAULT_PRICING))


@dataclass(frozen=True, slots=True)
class CostEntry:
    step: str
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    dbus: float
    usd: float


@dataclass(frozen=True, slots=True)
class CostSummary:
    calls: int
    prompt_tokens: int
    completion_tokens: int
    dbus: float
    usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _summarize(entries: list[CostEntry]) -> CostSummary:
    return CostSummary(
        calls=len(entries),
        prompt_tokens=sum(e.prompt_tokens for e in entries),
        completion_tokens=sum(e.completion_tokens for e in entries),
        dbus=round(sum(e.dbus for e in entries), 6),
        usd=round(sum(e.usd for e in entries), 6),
    )


class TokenCostTracker:
    """Thread-safe accumulator of LLM usage per step and endpoint."""

    def __init__(self, pricing: PricingTable | None = None, *, metrics: Metrics | None = None) -> None:
        self._pricing = pricing or PricingTable.default()
        self._metrics = metrics
        self._entries: list[CostEntry] = []
        self._lock = threading.Lock()

    def record(self, step: str, endpoint: str, usage: LLMUsage) -> CostEntry:
        if usage.prompt_tokens < 0 or usage.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        pricing = self._pricing.for_endpoint(endpoint)
        dbus = pricing.dbus(usage)
        entry = CostEntry(
            step=step,
            endpoint=endpoint,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            dbus=dbus,
            usd=dbus * self._pricing.usd_per_dbu,
        )
        with self._lock:
            self._entries.append(entry)
        if self._metrics is not None:
            self._metrics.increment("llm.tokens.prompt", usage.prompt_tokens, endpoint=endpoint, step=step)
            self._metrics.increment(
                "llm.tokens.completion", usage.completion_tokens, endpoint=endpoint, step=step
            )
            self._metrics.increment("llm.cost.usd", entry.usd, endpoint=endpoint)
        return entry

    def estimate(self, endpoint: str, usage: LLMUsage) -> float:
        """USD estimate for a prospective call (used with RunBudget.reserve)."""
        return self._pricing.for_endpoint(endpoint).dbus(usage) * self._pricing.usd_per_dbu

    def entries(self) -> list[CostEntry]:
        with self._lock:
            return list(self._entries)

    def totals(self) -> CostSummary:
        return _summarize(self.entries())

    def by_step(self) -> dict[str, CostSummary]:
        return self._group(lambda e: e.step)

    def by_endpoint(self) -> dict[str, CostSummary]:
        return self._group(lambda e: e.endpoint)

    def _group(self, key: Callable[[CostEntry], str]) -> dict[str, CostSummary]:
        groups: dict[str, list[CostEntry]] = {}
        for entry in self.entries():
            groups.setdefault(key(entry), []).append(entry)
        return {name: _summarize(items) for name, items in sorted(groups.items())}

    def as_metrics(self) -> dict[str, float]:
        """Flat numeric dict suitable for ``mlflow.log_metrics``."""
        totals = self.totals()
        data: dict[str, float] = {
            "llm_calls": float(totals.calls),
            "llm_prompt_tokens": float(totals.prompt_tokens),
            "llm_completion_tokens": float(totals.completion_tokens),
            "llm_total_tokens": float(totals.total_tokens),
            "llm_cost_dbu": totals.dbus,
            "llm_cost_usd": totals.usd,
        }
        for step, summary in self.by_step().items():
            safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in step)
            data[f"llm_tokens.{safe}"] = float(summary.total_tokens)
        return data

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
