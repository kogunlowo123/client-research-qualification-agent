"""Trigger and monitor Databricks Workflows job runs (used by CI smoke tests).

Thin wrapper over ``WorkspaceClient.jobs.run_now`` / ``get_run`` with explicit
polling so callers control the timeout, poll interval and cancellation policy,
and get a typed outcome instead of SDK objects.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from client_research_agent.databricks.errors import map_databricks_error
from client_research_agent.observability.logging import get_logger
from client_research_agent.utils.errors import AgentError, UpstreamTimeoutError

_log = get_logger(__name__)

TERMINAL_LIFECYCLE_STATES = frozenset({"TERMINATED", "SKIPPED", "INTERNAL_ERROR"})
SUCCESS_RESULT_STATES = frozenset({"SUCCESS", "SUCCESS_WITH_FAILURES"})


class JobRunFailedError(AgentError):
    def __init__(self, outcome: JobRunOutcome) -> None:
        super().__init__(
            f"job run {outcome.run_id} ended {outcome.life_cycle_state}/{outcome.result_state}: "
            f"{outcome.state_message or 'no message'} ({outcome.run_page_url or 'no url'})"
        )
        self.outcome = outcome


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


@dataclass(frozen=True, slots=True)
class JobRunOutcome:
    run_id: int
    life_cycle_state: str | None
    result_state: str | None
    state_message: str | None
    run_page_url: str | None

    @property
    def terminal(self) -> bool:
        return self.life_cycle_state in TERMINAL_LIFECYCLE_STATES

    @property
    def succeeded(self) -> bool:
        return self.terminal and self.result_state in SUCCESS_RESULT_STATES

    def raise_for_failure(self) -> JobRunOutcome:
        if not self.succeeded:
            raise JobRunFailedError(self)
        return self


class JobRunner:
    """``jobs_api`` is ``WorkspaceClient.jobs``."""

    def __init__(
        self,
        jobs_api: Any,
        *,
        poll_interval_seconds: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._jobs = jobs_api
        self._poll_interval = poll_interval_seconds
        self._sleep = sleep
        self._clock = clock

    def trigger(
        self,
        job_id: int,
        *,
        job_parameters: Mapping[str, str] | None = None,
        idempotency_token: str | None = None,
    ) -> int:
        """Start a run and return its ``run_id`` without waiting."""
        try:
            waiter = self._jobs.run_now(
                job_id=job_id,
                job_parameters=dict(job_parameters) if job_parameters else None,
                idempotency_token=idempotency_token,
            )
        except Exception as exc:
            raise map_databricks_error(exc, f"jobs.run_now({job_id})") from exc
        run_id = getattr(getattr(waiter, "response", None), "run_id", None)
        if run_id is None:
            run_id = waiter.run_id
        _log.info("job.triggered", job_id=job_id, run_id=run_id)
        return int(run_id)

    def status(self, run_id: int) -> JobRunOutcome:
        try:
            run = self._jobs.get_run(run_id=run_id)
        except Exception as exc:
            raise map_databricks_error(exc, f"jobs.get_run({run_id})") from exc
        state = getattr(run, "state", None)
        return JobRunOutcome(
            run_id=run_id,
            life_cycle_state=_enum_value(getattr(state, "life_cycle_state", None)),
            result_state=_enum_value(getattr(state, "result_state", None)),
            state_message=getattr(state, "state_message", None),
            run_page_url=getattr(run, "run_page_url", None),
        )

    def wait(self, run_id: int, *, timeout_seconds: float, cancel_on_timeout: bool = True) -> JobRunOutcome:
        """Poll until the run is terminal; on timeout optionally cancel and raise."""
        deadline = self._clock() + timeout_seconds
        while True:
            outcome = self.status(run_id)
            if outcome.terminal:
                _log.info(
                    "job.finished",
                    run_id=run_id,
                    life_cycle_state=outcome.life_cycle_state,
                    result_state=outcome.result_state,
                )
                return outcome
            if self._clock() >= deadline:
                if cancel_on_timeout:
                    try:
                        self._jobs.cancel_run(run_id=run_id)
                    except Exception as exc:
                        raise map_databricks_error(exc, f"jobs.cancel_run({run_id})") from exc
                raise UpstreamTimeoutError(
                    f"job run {run_id} still {outcome.life_cycle_state} after {timeout_seconds:.0f}s"
                )
            self._sleep(self._poll_interval)

    def run_and_wait(
        self,
        job_id: int,
        *,
        timeout_seconds: float,
        job_parameters: Mapping[str, str] | None = None,
        idempotency_token: str | None = None,
        raise_on_failure: bool = True,
    ) -> JobRunOutcome:
        run_id = self.trigger(job_id, job_parameters=job_parameters, idempotency_token=idempotency_token)
        outcome = self.wait(run_id, timeout_seconds=timeout_seconds)
        return outcome.raise_for_failure() if raise_on_failure else outcome
