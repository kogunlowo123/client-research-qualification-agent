"""Run state: what every orchestration step did, how long it took and what it produced."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from client_research_agent.models import ClientBrief, ResearchRequest
from client_research_agent.models.domain import utc_now

MAX_ERROR_CHARS = 300


class StepStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepName(StrEnum):
    COMPANY_INPUT = "company_input"
    RESEARCH_PLAN = "research_plan"
    EVIDENCE_GATHERING = "evidence_gathering"
    RETRIEVAL = "retrieval"
    QUALIFICATION = "qualification"
    SCORING = "scoring"
    BRIEF_GENERATION = "brief_generation"
    VALIDATION = "validation"
    OUTPUT = "output"


@dataclass
class StepRecord:
    name: StepName
    started_at: datetime = field(default_factory=utc_now)
    status: StepStatus = StepStatus.OK
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def degrade(self, warning: str) -> None:
        """Record a recoverable problem; a skipped or failed step keeps its status."""
        if self.status is StepStatus.OK:
            self.status = StepStatus.DEGRADED
        self.warnings.append(warning)

    def note(self, warning: str) -> None:
        """Record an informational warning (surfaced in the brief) without changing the status."""
        self.warnings.append(warning)

    def skip(self, reason: str) -> None:
        self.status = StepStatus.SKIPPED
        self.detail["skip_reason"] = reason

    def fail(self, exc: BaseException) -> None:
        self.status = StepStatus.FAILED
        self.error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 3),
            "warnings": list(self.warnings),
            "detail": dict(self.detail),
            "error": self.error,
        }


@dataclass
class RunState:
    run_id: str
    request: ResearchRequest
    principal: str
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    steps: list[StepRecord] = field(default_factory=list)
    brief: ClientBrief | None = None
    mlflow_run_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def step(self, name: StepName) -> StepRecord | None:
        return next((s for s in self.steps if s.name is name), None)

    @property
    def status(self) -> StepStatus:
        statuses = {s.status for s in self.steps}
        if StepStatus.FAILED in statuses:
            return StepStatus.FAILED
        if StepStatus.DEGRADED in statuses:
            return StepStatus.DEGRADED
        return StepStatus.OK

    @property
    def warnings(self) -> list[str]:
        return list(dict.fromkeys(w for s in self.steps for w in s.warnings))

    @property
    def degraded_steps(self) -> list[str]:
        return [s.name.value for s in self.steps if s.status in (StepStatus.DEGRADED, StepStatus.FAILED)]

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "company": self.request.company_name,
            "principal": self.principal,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": round(self.duration_ms, 3),
            "mlflow_run_id": self.mlflow_run_id,
            "metrics": dict(self.metrics),
            "steps": [s.to_dict() for s in self.steps],
        }
