"""Deterministic brief-quality metrics.

All metrics are computed from the :class:`ClientBrief` alone (plus optional
labels), so they are reproducible, cheap and usable both offline and as MLflow
custom scorers.

``citation_coverage``
    Share of fact statements the citation validator supported. A brief that
    asserts no facts has nothing uncited, so its coverage is 1.0.
``grounded_fact_ratio``
    Share of verified facts whose text is lexically supported
    (:func:`~client_research_agent.citations.entailment.lexical_support`) by the
    quotes of the evidence they cite.
``verdict_accuracy``
    1.0 when the verdict matches the label, 0.0 otherwise, ``None`` unlabelled.
``criterion_score_mae``
    Mean absolute error of criterion scores against labelled scores.
``section_completeness``
    Share of the brief's sections that carry at least one statement.
``discovery_questions_ok``
    1.0 when the brief has exactly five discovery questions.
``expected_fact_recall``
    Share of labelled expected facts supported by the brief's text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from client_research_agent.citations.entailment import lexical_support
from client_research_agent.evaluation.dataset import EvalExpectations
from client_research_agent.models import ClientBrief, Criterion, FitVerdict

REQUIRED_DISCOVERY_QUESTIONS = 5
DEFAULT_SUPPORT_THRESHOLD = 0.3
SECTIONS: tuple[str, ...] = (
    "company_overview",
    "technology_priorities",
    "gartner_relevant_insights",
    "opportunities",
    "risks",
    "executive_summary",
    "executive_talking_points",
    "recommended_next_actions",
)


def citation_coverage(brief: ClientBrief) -> float:
    report = brief.citation_report
    return report.coverage if report.total_statements else 1.0


def grounded_fact_ratio(brief: ClientBrief, *, threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> float:
    facts = brief.verified_facts
    if not facts:
        return 1.0
    quotes = {e.evidence_id: e.quote for e in brief.evidence}
    grounded = 0
    for fact in facts:
        text = "\n".join(quotes[i] for i in fact.evidence_ids if i in quotes)
        if text and lexical_support(fact.text, text).score >= threshold:
            grounded += 1
    return grounded / len(facts)


def verdict_accuracy(brief: ClientBrief, expected: FitVerdict | None) -> float | None:
    if expected is None:
        return None
    return 1.0 if brief.qualification.verdict is expected else 0.0


def criterion_score_mae(brief: ClientBrief, expected: Mapping[Criterion, int]) -> float | None:
    if not expected:
        return None
    actual = {s.criterion: s.score for s in brief.qualification.scores}
    errors = [abs(actual.get(criterion, 0) - score) for criterion, score in expected.items()]
    return sum(errors) / len(errors)


def section_completeness(brief: ClientBrief) -> float:
    filled = sum(1 for name in SECTIONS if getattr(brief, name))
    return filled / len(SECTIONS)


def discovery_questions_ok(brief: ClientBrief) -> float:
    return 1.0 if len(brief.discovery_questions) == REQUIRED_DISCOVERY_QUESTIONS else 0.0


def brief_text(brief: ClientBrief) -> str:
    parts = [s.text for s in brief.all_statements()]
    parts.extend(e.quote for e in brief.evidence)
    parts.extend(brief.discovery_questions)
    return "\n".join(parts)


def expected_fact_recall(
    brief: ClientBrief, facts: Sequence[str], *, threshold: float = DEFAULT_SUPPORT_THRESHOLD
) -> float | None:
    if not facts:
        return None
    text = brief_text(brief)
    found = sum(1 for fact in facts if lexical_support(fact, text).score >= threshold)
    return found / len(facts)


@dataclass(frozen=True, slots=True)
class BriefMetrics:
    citation_coverage: float
    grounded_fact_ratio: float
    section_completeness: float
    discovery_questions_ok: float
    verdict_accuracy: float | None = None
    criterion_score_mae: float | None = None
    expected_fact_recall: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items() if v is not None}


def compute_metrics(
    brief: ClientBrief,
    expectations: EvalExpectations | None = None,
    *,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> BriefMetrics:
    labels = expectations or EvalExpectations()
    return BriefMetrics(
        citation_coverage=citation_coverage(brief),
        grounded_fact_ratio=grounded_fact_ratio(brief, threshold=support_threshold),
        section_completeness=section_completeness(brief),
        discovery_questions_ok=discovery_questions_ok(brief),
        verdict_accuracy=verdict_accuracy(brief, labels.expected_verdict),
        criterion_score_mae=criterion_score_mae(brief, labels.expected_scores),
        expected_fact_recall=expected_fact_recall(brief, labels.expected_facts, threshold=support_threshold),
    )


def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mean of every metric over the rows that report it."""
    totals: dict[str, list[float]] = {}
    for row in values:
        for key, value in row.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in sorted(totals.items())}
