"""ClientBrief builders shared by governance and security tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime

from client_research_agent.models import (
    BriefStatement,
    ClientBrief,
    Criterion,
    CriterionScore,
    DocumentType,
    Evidence,
    FitVerdict,
    ProvenanceKind,
    QualificationResult,
)

EVIDENCE_URL = "https://acme.example.com/news/cloud-migration"


def evidence(evidence_id: str = "ev-1", url: str = EVIDENCE_URL, chunk_id: str = "chunk-1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        url=url,
        title="Acme completes cloud migration",
        quote="Acme migrated its ERP platform to the cloud in 2025.",
        document_type=DocumentType.PRESS_RELEASE,
        publication_date=date(2025, 11, 3),
        relevance=0.9,
    )


def fact(text: str, *evidence_ids: str) -> BriefStatement:
    return BriefStatement(
        text=text, provenance=ProvenanceKind.VERIFIED_FACT, evidence_ids=evidence_ids or ("ev-1",)
    )


def recommendation(text: str) -> BriefStatement:
    return BriefStatement(text=text, provenance=ProvenanceKind.AI_RECOMMENDATION)


def build_brief(
    *,
    overview: Sequence[BriefStatement] | None = None,
    opportunities: Sequence[BriefStatement] | None = None,
    next_actions: Sequence[BriefStatement] | None = None,
    risks: Sequence[BriefStatement] | None = None,
    questions: Sequence[str] | None = None,
    evidence_items: Sequence[Evidence] | None = None,
    warnings: Sequence[str] = (),
    rationale: str = "Strong modernization signals backed by cited evidence.",
) -> ClientBrief:
    score = CriterionScore(
        criterion=Criterion.TECH_MODERNIZATION,
        score=4,
        weight=0.2,
        confidence=0.8,
        rationale="ERP moved to the cloud.",
        evidence_ids=("ev-1",),
    )
    return ClientBrief(
        run_id="run-123",
        company="Acme Corp",
        generated_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
        company_overview=tuple(overview or (fact("Acme migrated its ERP platform to the cloud in 2025."),)),
        qualification=QualificationResult(
            scores=(score,),
            weighted_score=3.8,
            overall_confidence=0.75,
            verdict=FitVerdict.GOOD_FIT,
            verdict_rationale=rationale,
        ),
        evidence=tuple(evidence_items or (evidence(),)),
        technology_priorities=(fact("Acme is consolidating data platforms."),),
        gartner_relevant_insights=(recommendation("Data platform consolidation is a common 2026 priority."),),
        opportunities=tuple(opportunities or (recommendation("Offer a lakehouse governance assessment."),)),
        risks=tuple(risks or (recommendation("Budget cycles may delay new projects."),)),
        executive_summary=(fact("Acme is modernizing its core systems."),),
        discovery_questions=tuple(
            questions
            or (
                "What are your data platform priorities for 2027?",
                "How is the ERP migration progressing?",
                "Who owns data governance today?",
                "What AI use cases are funded?",
                "What is your timeline for vendor selection?",
            )
        ),
        executive_talking_points=(recommendation("Lead with governance outcomes."),),
        recommended_next_actions=tuple(next_actions or (recommendation("Schedule a discovery workshop."),)),
        warnings=tuple(warnings),
    )
