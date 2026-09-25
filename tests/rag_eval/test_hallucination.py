"""Generation quality gate: fabricated facts must never survive into a Client Brief as verified facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from client_research_agent.briefing.generator import BriefGenerationAgent
from client_research_agent.citations.entailment import lexical_support
from client_research_agent.config.settings import AppSettings
from client_research_agent.models import ClientBrief, ProvenanceKind
from client_research_agent.prompts.registry import default_registry
from client_research_agent.qualification.agent import QualificationAgent
from client_research_agent.qualification.text import unmatched_numbers
from client_research_agent.scoring.engine import ScoringEngine
from client_research_agent.services.ports import ChatMessage
from tests.support.doubles import ScriptedLLM
from tests.unit.qualification.helpers import COMPANY, TODAY, KeywordRetriever, acme_corpus

pytestmark = pytest.mark.rag_eval

FABRICATIONS = (
    "Acme Corp reported annual revenue of $31.7 billion for fiscal 2025.",
    "The company employs approximately 450,000 employees in 30 countries.",
    "Acme acquired Globex for $4 billion in 2026.",
    "Acme Corp is ranked first in the Gartner Magic Quadrant for data platforms.",
    "The cloud migration will complete by the end of 2031.",
)
GROUNDED = (
    "Acme Corp reported annual revenue of $12.4 billion for fiscal 2025.",
    "A chief data officer was appointed in March 2026 to lead data governance.",
)


def _hallucinating_writer(messages: Sequence[ChatMessage]) -> Mapping[str, Any]:
    ids = [f"E{i}" for i in range(1, 6)]
    facts = [
        {"text": text, "provenance": "verified_fact", "evidence_ids": ids}
        for text in (*FABRICATIONS[:3], *GROUNDED)
    ]
    return {
        "company_overview": facts,
        "executive_summary": [
            {"text": FABRICATIONS[3], "provenance": "verified_fact", "evidence_ids": ["E1"]},
            {"text": "Acme looks like a strong prospect.", "provenance": "ai_recommendation"},
        ],
        "executive_talking_points": [
            {"text": FABRICATIONS[4], "provenance": "verified_fact", "evidence_ids": ["E2", "E3"]},
            {"text": "Revenue is $12.4 billion.", "provenance": "verified_fact", "evidence_ids": ["E404"]},
        ],
        "recommended_next_actions": [
            {"text": "Schedule a discovery call.", "provenance": "ai_recommendation"}
        ],
    }


def _qualifier_with_fake_ids(messages: Sequence[ChatMessage]) -> Mapping[str, Any]:
    return {
        "score": 5,
        "confidence": 0.99,
        "rationale": "Acme has $90 billion revenue.",
        "evidence_ids": ["E900", "E901"],
        "reasoning_steps": ["Invented evidence"],
    }


@pytest.fixture
def brief(settings: AppSettings) -> ClientBrief:
    registry = default_registry()
    qualifier_llm = ScriptedLLM(routes={"ONE criterion:": _qualifier_with_fake_ids})
    output = QualificationAgent(
        qualifier_llm, KeywordRetriever(acme_corpus()), registry, settings, today=TODAY
    ).qualify(COMPANY)
    result = ScoringEngine(settings.scoring).evaluate(output.scores)
    writer_llm = ScriptedLLM(
        routes={
            "strict fact-checking judge": {"supported": True, "score": 1.0, "reason": "Trust me, supported."},
            "executive client brief": _hallucinating_writer,
            "opportunity analysis section of a client brief": {
                "technology_priorities": [
                    {
                        "text": "Acme will spend $2 billion on AI in 2027.",
                        "provenance": "verified_fact",
                        "evidence_ids": ["E1", "E2", "E3"],
                    }
                ],
                "gartner_insights": [
                    {
                        "theme": "AI engineering",
                        "text": "Gartner predicts Acme will win.",
                        "evidence_ids": ["E3"],
                    }
                ],
            },
        }
    )
    return BriefGenerationAgent(writer_llm, registry, settings).generate(
        run_id="hallucination-eval",
        company=COMPANY,
        qualification=result,
        evidence=output.evidence,
        warnings=output.warnings,
    )


def test_no_fabricated_statement_is_a_verified_fact(brief: ClientBrief) -> None:
    fact_texts = {s.text for s in brief.verified_facts}
    for fabricated in (*FABRICATIONS, "Acme will spend $2 billion on AI in 2027."):
        assert fabricated not in fact_texts


def test_fabricated_numbers_are_removed_not_relabelled(brief: ClientBrief) -> None:
    all_text = " ".join(s.text for s in brief.all_statements())
    for figure in ("$31.7 billion", "450,000", "$4 billion", "2031", "$2 billion"):
        assert figure not in all_text
    removed = set(brief.citation_report.removed_statements)
    assert set(FABRICATIONS[:3]) <= removed
    assert FABRICATIONS[4] in removed


def test_grounded_facts_survive(brief: ClientBrief) -> None:
    fact_texts = {s.text for s in brief.verified_facts}
    assert set(GROUNDED) <= fact_texts


def test_every_fact_is_entailed_by_its_cited_evidence(brief: ClientBrief, settings: AppSettings) -> None:
    evidence = {e.evidence_id: e for e in brief.evidence}
    assert brief.verified_facts
    for statement in brief.verified_facts:
        assert statement.provenance is ProvenanceKind.VERIFIED_FACT
        assert statement.evidence_ids
        best = 0.0
        for evidence_id in statement.evidence_ids:
            item = evidence[evidence_id]
            best = max(best, lexical_support(statement.text, f"{item.title}\n{item.quote}").score)
        # Quotes are a subset of the chunk; facts are verified against the full chunk, so allow
        # quote-level support to be lower but still require every figure to be present somewhere cited.
        assert best > 0.0
        cited_text = " ".join(evidence[i].quote for i in statement.evidence_ids)
        assert (
            not unmatched_numbers(statement.text, cited_text)
            or best >= settings.guardrails.min_citation_support
        )


def test_llm_scores_citing_invented_evidence_are_discarded(brief: ClientBrief) -> None:
    for score in brief.qualification.scores:
        assert all(i in {e.evidence_id for e in brief.evidence} for i in score.evidence_ids)
        assert "$90 billion" not in score.rationale
    assert any("not grounded" in w for w in brief.warnings)


def test_gartner_is_never_misattributed(brief: ClientBrief) -> None:
    for statement in brief.gartner_relevant_insights:
        assert "Gartner predicts" not in statement.text
        assert statement.provenance is ProvenanceKind.AI_RECOMMENDATION
    assert any("attributing a claim to Gartner" in w for w in brief.warnings)
    assert not any("Magic Quadrant" in s.text for s in brief.verified_facts)
