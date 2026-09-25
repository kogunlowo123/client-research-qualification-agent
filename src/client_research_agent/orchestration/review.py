"""Human-in-the-loop review and the analyst feedback loop.

:class:`ReviewPolicy`
    Decides whether a brief must be reviewed by an analyst before it is used:
    NOT_ENOUGH_EVIDENCE verdicts, citation coverage below the threshold, output
    guard violations, responsible-AI findings, a verdict that a single +/-1
    criterion change would flip (``ScoringEngine.sensitivity``) and runs in
    which a pipeline step degraded.

:class:`ReviewQueue`
    Append-only JSON Lines log of review events. The current state of an item
    is its latest event, so concurrent writers never rewrite history.

:class:`FeedbackStore`
    Records analyst feedback (verdict override, statement corrections, a 1-5
    rating) and exports it as evaluation examples, closing the loop between
    production reviews and the regression suite.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from client_research_agent.evaluation.dataset import EvalExample, EvalExpectations, EvalInputs
from client_research_agent.governance.responsible_ai import PolicyReport, Severity
from client_research_agent.models import ClientBrief, FitVerdict, ResearchRequest
from client_research_agent.models.domain import utc_now
from client_research_agent.retrieval.self_rag import AnswerCritique, SupportLevel
from client_research_agent.scoring.engine import SensitivityReport
from client_research_agent.security.output_guard import OutputReport


class ReviewReason(StrEnum):
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    LOW_CITATION_COVERAGE = "low_citation_coverage"
    OUTPUT_GUARD = "output_guard_violation"
    POLICY_FINDING = "responsible_ai_finding"
    VERDICT_SENSITIVE = "verdict_sensitive"
    DEGRADED_RUN = "degraded_run"
    UNSUPPORTED_SUMMARY = "unsupported_summary"


class ReviewPriority(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


_HIGH = frozenset({ReviewReason.OUTPUT_GUARD, ReviewReason.POLICY_FINDING})
_MEDIUM = frozenset(
    {ReviewReason.LOW_CITATION_COVERAGE, ReviewReason.VERDICT_SENSITIVE, ReviewReason.UNSUPPORTED_SUMMARY}
)


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    needs_review: bool
    reasons: tuple[ReviewReason, ...]
    details: tuple[str, ...]
    priority: ReviewPriority

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_review": self.needs_review,
            "reasons": [r.value for r in self.reasons],
            "details": list(self.details),
            "priority": self.priority.value,
        }


@dataclass(frozen=True, slots=True)
class ReviewPolicy:
    min_citation_coverage: float = 0.8
    review_not_enough_evidence: bool = True
    review_sensitive_verdicts: bool = True
    review_degraded_runs: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_citation_coverage <= 1.0:
            raise ValueError("min_citation_coverage must be within [0, 1]")

    def decide(
        self,
        brief: ClientBrief,
        *,
        sensitivity: SensitivityReport | None = None,
        output_report: OutputReport | None = None,
        policy_report: PolicyReport | None = None,
        degraded_steps: Sequence[str] = (),
        reflection: AnswerCritique | None = None,
    ) -> ReviewDecision:
        reasons: list[ReviewReason] = []
        details: list[str] = []
        verdict = brief.qualification.verdict
        if self.review_not_enough_evidence and verdict is FitVerdict.NOT_ENOUGH_EVIDENCE:
            reasons.append(ReviewReason.NOT_ENOUGH_EVIDENCE)
            details.append("verdict is not_enough_evidence; confirm before disqualifying the account")
        report = brief.citation_report
        if report.total_statements and report.coverage < self.min_citation_coverage:
            reasons.append(ReviewReason.LOW_CITATION_COVERAGE)
            details.append(
                f"citation coverage {report.coverage:.0%} is below the "
                f"{self.min_citation_coverage:.0%} minimum"
            )
        if output_report is not None and not output_report.ok:
            reasons.append(ReviewReason.OUTPUT_GUARD)
            kinds = ", ".join(sorted(k.value for k in output_report.kinds))
            details.append(f"output guard violations: {kinds}")
        if policy_report is not None and policy_report.findings:
            reasons.append(ReviewReason.POLICY_FINDING)
            severities = sorted({f.severity.value for f in policy_report.findings})
            details.append(
                f"{len(policy_report.findings)} responsible-AI finding(s) ({', '.join(severities)})"
            )
        if self.review_sensitive_verdicts and sensitivity is not None and not sensitivity.robust:
            reasons.append(ReviewReason.VERDICT_SENSITIVE)
            details.append(sensitivity.summary())
        if reflection is not None and reflection.is_supported is SupportLevel.NO:
            reasons.append(ReviewReason.UNSUPPORTED_SUMMARY)
            details.append(
                f"self-reflection: executive summary not supported by cited evidence ({reflection.reason})"
            )
        if self.review_degraded_runs and degraded_steps:
            reasons.append(ReviewReason.DEGRADED_RUN)
            details.append(f"degraded steps: {', '.join(degraded_steps)}")
        return ReviewDecision(
            needs_review=bool(reasons),
            reasons=tuple(reasons),
            details=tuple(details),
            priority=_priority(reasons, policy_report),
        )


def _priority(reasons: Sequence[ReviewReason], policy_report: PolicyReport | None) -> ReviewPriority:
    if not reasons:
        return ReviewPriority.NONE
    errors = policy_report is not None and any(f.severity is Severity.ERROR for f in policy_report.findings)
    if errors or any(r in _HIGH for r in reasons):
        return ReviewPriority.HIGH
    if any(r in _MEDIUM for r in reasons):
        return ReviewPriority.MEDIUM
    return ReviewPriority.LOW


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewItem(_Record):
    run_id: str
    company: str
    domain: str | None = None
    ticker: str | None = None
    verdict: FitVerdict
    weighted_score: float
    citation_coverage: float
    reasons: tuple[ReviewReason, ...]
    details: tuple[str, ...] = ()
    priority: ReviewPriority
    requested_by: str = "system"
    status: ReviewStatus = ReviewStatus.PENDING
    enqueued_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None
    reviewer: str | None = None
    notes: str = ""


class _JsonlLog:
    """Thread-safe append-only JSON Lines file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), sort_keys=True, ensure_ascii=False, default=str)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._path.exists():
                return []
            lines = self._path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


class ReviewQueue:
    """Analyst review queue persisted as an append-only event log."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._log = _JsonlLog(path)

    @property
    def path(self) -> Path:
        return self._log.path

    def enqueue(
        self,
        brief: ClientBrief,
        decision: ReviewDecision,
        *,
        request: ResearchRequest | None = None,
    ) -> ReviewItem:
        if not decision.needs_review:
            raise ValueError("decision does not require review")
        item = ReviewItem(
            run_id=brief.run_id,
            company=brief.company,
            domain=request.domain if request is not None else None,
            ticker=request.ticker if request is not None else None,
            verdict=brief.qualification.verdict,
            weighted_score=brief.qualification.weighted_score,
            citation_coverage=brief.citation_report.coverage,
            reasons=decision.reasons,
            details=decision.details,
            priority=decision.priority,
            requested_by=request.requested_by if request is not None else "system",
        )
        self._log.append(item.model_dump(mode="json"))
        return item

    def items(self) -> list[ReviewItem]:
        """Latest state of every item, in first-enqueued order."""
        latest: dict[str, ReviewItem] = {}
        for record in self._log.read():
            item = ReviewItem.model_validate(record)
            latest[item.run_id] = item
        return list(latest.values())

    def get(self, run_id: str) -> ReviewItem | None:
        return next((item for item in self.items() if item.run_id == run_id), None)

    def pending(self) -> list[ReviewItem]:
        order = {
            ReviewPriority.HIGH: 0,
            ReviewPriority.MEDIUM: 1,
            ReviewPriority.LOW: 2,
            ReviewPriority.NONE: 3,
        }
        waiting = [item for item in self.items() if item.status is ReviewStatus.PENDING]
        return sorted(waiting, key=lambda item: (order[item.priority], item.enqueued_at))

    def resolve(self, run_id: str, *, reviewer: str, approved: bool, notes: str = "") -> ReviewItem:
        current = self.get(run_id)
        if current is None:
            raise KeyError(f"no review item for run {run_id!r}")
        if current.status is not ReviewStatus.PENDING:
            raise ValueError(f"review for run {run_id!r} is already {current.status.value}")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        resolved = current.model_copy(
            update={
                "status": ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED,
                "resolved_at": utc_now(),
                "reviewer": reviewer,
                "notes": notes,
            }
        )
        self._log.append(resolved.model_dump(mode="json"))
        return resolved


class StatementCorrection(_Record):
    original: str = Field(min_length=1)
    corrected: str | None = None
    reason: str = ""


class AnalystFeedback(_Record):
    feedback_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    verdict_override: FitVerdict | None = None
    corrections: tuple[StatementCorrection, ...] = ()
    rating: int | None = Field(default=None, ge=1, le=5)
    comments: str = ""
    created_at: datetime = Field(default_factory=utc_now)


#: Ratings at or above this mark a brief's verdict and facts as analyst-approved ground truth.
TRUSTED_RATING = 4
MAX_EXPORTED_FACTS = 8


class FeedbackStore:
    """Analyst feedback, exportable as evaluation examples (the feedback loop)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._log = _JsonlLog(path)

    @property
    def path(self) -> Path:
        return self._log.path

    def record(self, feedback: AnalystFeedback) -> AnalystFeedback:
        self._log.append(feedback.model_dump(mode="json"))
        return feedback

    def entries(self, run_id: str | None = None) -> list[AnalystFeedback]:
        entries = [AnalystFeedback.model_validate(r) for r in self._log.read()]
        return [e for e in entries if run_id is None or e.run_id == run_id]

    def export_eval_examples(
        self,
        brief_lookup: Callable[[str], ClientBrief | None],
        *,
        review_queue: ReviewQueue | None = None,
    ) -> list[EvalExample]:
        """One example per reviewed run; the latest feedback on a run wins."""
        latest: dict[str, AnalystFeedback] = {}
        for entry in self.entries():
            latest[entry.run_id] = entry
        examples: list[EvalExample] = []
        for run_id, feedback in latest.items():
            brief = brief_lookup(run_id)
            if brief is None:
                continue
            example = feedback_to_example(
                feedback, brief, review_queue.get(run_id) if review_queue is not None else None
            )
            if example is not None:
                examples.append(example)
        return examples


def feedback_to_example(
    feedback: AnalystFeedback, brief: ClientBrief, review: ReviewItem | None = None
) -> EvalExample | None:
    trusted = feedback.rating is not None and feedback.rating >= TRUSTED_RATING
    verdict = feedback.verdict_override or (brief.qualification.verdict if trusted else None)
    removed = {c.original.strip() for c in feedback.corrections}
    corrected = [c.corrected.strip() for c in feedback.corrections if c.corrected and c.corrected.strip()]
    approved: Iterable[str] = (
        (s.text for s in brief.verified_facts if s.text.strip() not in removed) if trusted else ()
    )
    facts = tuple(dict.fromkeys([*corrected, *approved]))[:MAX_EXPORTED_FACTS]
    expectations = EvalExpectations(expected_verdict=verdict, expected_facts=facts)
    if expectations.is_empty:
        return None
    tags = {"source": "analyst_feedback", "reviewer": feedback.reviewer, "run_id": brief.run_id}
    if feedback.rating is not None:
        tags["rating"] = str(feedback.rating)
    return EvalExample(
        eval_id=f"feedback-{feedback.feedback_id}",
        inputs=EvalInputs(
            company_name=brief.company,
            domain=review.domain if review is not None else None,
            ticker=review.ticker if review is not None else None,
        ),
        expectations=expectations,
        tags=tags,
    )
