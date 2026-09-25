"""Per-run LLM metering: token/cost accounting and hard budget enforcement.

Every LLM call made by any agent goes through :class:`MeteredLLMClient`. While
a run is active (see :func:`run_accounting`), each call is

1. *reserved* against the run's :class:`~client_research_agent.security.RunBudget`
   using an estimate of the prompt size plus half of ``max_tokens``;
2. executed on the wrapped client;
3. *charged* with the actual usage and recorded in the run's
   :class:`~client_research_agent.observability.cost.TokenCostTracker` under the
   current orchestration step.

When the budget is exhausted the client raises :class:`LLMBudgetExhaustedError`,
a :class:`~client_research_agent.utils.errors.CircuitOpenError`, which every
agent already treats as "LLM unavailable" and answers deterministically. The
accounting lives in a context variable, so worker threads that copy the context
(qualification, multi-query retrieval) are metered against the same run.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from client_research_agent.observability.cost import TokenCostTracker
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.security.rate_limiter import BudgetExceededError, RunBudget
from client_research_agent.services.ports import ChatMessage, LLMClient, LLMResponse, LLMUsage
from client_research_agent.utils.errors import CircuitOpenError

CHARS_PER_TOKEN = 4
_log = get_logger(__name__)


class LLMBudgetExhaustedError(CircuitOpenError):
    """The run's token/cost/call budget is spent; callers fall back to deterministic logic."""

    def __init__(self, detail: str) -> None:
        super().__init__("llm_budget", 0.0)
        self.detail = detail


@dataclass
class RunAccounting:
    """Mutable per-run accounting shared (by reference) with worker threads."""

    budget: RunBudget
    costs: TokenCostTracker = field(default_factory=TokenCostTracker)
    step: str = "unassigned"
    exhausted: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_exhausted(self) -> bool:
        """Flag the budget as spent; returns True only for the first caller."""
        with self._lock:
            first = not self.exhausted
            self.exhausted = True
            return first


_CURRENT: ContextVar[RunAccounting | None] = ContextVar("cra_run_accounting", default=None)


@contextmanager
def run_accounting(accounting: RunAccounting) -> Iterator[RunAccounting]:
    token = _CURRENT.set(accounting)
    try:
        yield accounting
    finally:
        _CURRENT.reset(token)


def current_accounting() -> RunAccounting | None:
    return _CURRENT.get()


def estimate_prompt_tokens(messages: Sequence[ChatMessage]) -> int:
    return max(1, sum(len(m.content) for m in messages) // CHARS_PER_TOKEN)


class MeteredLLMClient:
    """``LLMClient`` decorator that enforces the active run's budget and records cost."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @property
    def inner(self) -> LLMClient:
        return self._inner

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        accounting = current_accounting()
        if accounting is None:
            return self._inner.complete(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
            )
        if accounting.exhausted:
            raise LLMBudgetExhaustedError("run budget already exhausted")
        prompt_tokens = estimate_prompt_tokens(messages)
        estimate = LLMUsage(prompt_tokens=prompt_tokens, completion_tokens=max_tokens // 2)
        try:
            accounting.budget.reserve(
                estimate.total_tokens, accounting.costs.estimate(self.model_name, estimate)
            )
        except BudgetExceededError as exc:
            self._exhausted(accounting, exc)
            raise LLMBudgetExhaustedError(str(exc)) from exc

        response = self._inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        entry = accounting.costs.record(accounting.step, self.model_name, response.usage)
        try:
            accounting.budget.charge(response.usage, entry.usd)
        except BudgetExceededError as exc:
            # The call already happened; keep its answer but refuse any further calls.
            self._exhausted(accounting, exc)
        return response

    @staticmethod
    def _exhausted(accounting: RunAccounting, exc: BudgetExceededError) -> None:
        if accounting.mark_exhausted():
            get_metrics().increment("llm.budget_exhausted", step=accounting.step)
            _log.warning("llm.budget_exhausted", step=accounting.step, detail=str(exc)[:200])
