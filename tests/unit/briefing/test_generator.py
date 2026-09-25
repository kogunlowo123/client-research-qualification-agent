from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from client_research_agent.briefing.generator import (
    DISCOVERY_QUESTION_COUNT,
    BriefGenerationAgent,
    DeterministicBriefBuilder,
    merge_reports,
)
from client_research_agent.config.settings import AppSettings, ScoringSettings
from client_research_agent.models import CitationCheck, CitationReport, FitVerdict, ProvenanceKind
from client_research_agent.prompts.registry import default_registry
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.scoring.engine import ScoringEngine
from client_research_agent.services.ports import ChatMessage
from tests.support.doubles import ScriptedLLM
from tests.unit.briefing.conftest import Qualified
from tests.unit.qualification.helpers import COMPANY, uniform_scores

FACT = ProvenanceKind.VERIFIED_FACT
REC = ProvenanceKind.AI_RECOMMENDATION
OPPORTUNITY_KEY = "opportunity analysis section of a client brief"
WRITER_KEY = "executive client brief"
QUESTIONS_KEY = "first discovery conversation"
JUDGE_KEY = "strict fact-checking judge"


def id_of(qualified: Qualified, needle: str) -> str:
    return next(
        e.evidence_id for e in qualified.evidence if needle in qualified.evidence.source_text(e.evidence_id)
    )


def writer_reply(qualified: Qualified) -> Mapping[str, Any]:
    revenue = id_of(qualified, "$12.4 billion")
    cdo = id_of(qualified, "chief data officer")
    return {
        "company_overview": [
            {
                "text": "Acme Corp reported annual revenue of $12.4 billion for fiscal 2025.",
                "provenance": "verified_fact",
                "evidence_ids": [revenue],
            },
            {
                "text": "Acme Corp employs approximately 90,000 employees.",
                "provenance": "verified_fact",
                "evidence_ids": [revenue],
            },
            {"text": "Acme is an obvious buyer.", "provenance": "ai_recommendation", "evidence_ids": []},
        ],
        "executive_summary": [{"text": "Acme is a strong prospect.", "evidence_ids": [revenue, "E77"]}],
        "executive_talking_points": [
            {"text": f"Congratulate the new chief data officer [{cdo}].", "provenance": "ai_recommendation"}
        ],
        "recommended_next_actions": [
            {"text": "Book a discovery workshop.", "provenance": "ai_recommendation"}
        ],
    }


def questions_reply(questions: Sequence[str]) -> Mapping[str, Any]:
    return {"questions": list(questions)}


def test_deterministic_brief_is_complete(settings: AppSettings, qualified: Qualified) -> None:
    agent = BriefGenerationAgent(None, default_registry(), settings)
    brief = agent.generate(
        run_id="run-1",
        company=COMPANY,
        qualification=qualified.result,
        evidence=qualified.evidence,
        warnings=("upstream warning",),
        model_versions={"prompt:criterion_qualifier": "custom"},
    )
    assert brief.run_id == "run-1"
    assert len(brief.discovery_questions) == DISCOVERY_QUESTION_COUNT
    assert brief.company_overview
    assert all(s.provenance is FACT for s in brief.company_overview)
    for section in (
        brief.executive_summary,
        brief.technology_priorities,
        brief.gartner_relevant_insights,
        brief.opportunities,
        brief.risks,
        brief.executive_talking_points,
        brief.recommended_next_actions,
    ):
        assert section
    assert brief.citation_report.total_statements == len(brief.verified_facts)
    assert brief.citation_report.coverage == 1.0
    assert brief.model_versions["llm"] == "none"
    assert brief.model_versions["prompt:criterion_qualifier"] == "custom"
    assert "prompt:brief_writer" in brief.model_versions
    assert brief.warnings[0] == "upstream warning"
    cited = {i for s in brief.all_statements() for i in s.evidence_ids}
    assert cited <= {e.evidence_id for e in brief.evidence}


def test_llm_brief_is_validated(settings: AppSettings, qualified: Qualified) -> None:
    llm = ScriptedLLM(
        routes={
            JUDGE_KEY: {"supported": True, "score": 0.9, "reason": "stated"},
            QUESTIONS_KEY: questions_reply(
                [
                    "Which AI use cases are in production today?",
                    "Not a question",
                    "Which AI use cases are in production today?",
                    "Who owns the data platform budget?",
                ]
            ),
            OPPORTUNITY_KEY: {},
            WRITER_KEY: writer_reply(qualified),
        }
    )
    brief = BriefGenerationAgent(llm, default_registry(), settings).generate(
        run_id="run-2", company=COMPANY, qualification=qualified.result, evidence=qualified.evidence
    )
    overview = [s.text for s in brief.company_overview]
    assert overview == ["Acme Corp reported annual revenue of $12.4 billion for fiscal 2025."]
    assert "Acme Corp employs approximately 90,000 employees." in brief.citation_report.removed_statements
    assert brief.executive_summary[0].evidence_ids == (id_of(qualified, "$12.4 billion"),)
    assert brief.executive_talking_points[0].text == "Congratulate the new chief data officer."
    assert brief.discovery_questions[:2] == (
        "Which AI use cases are in production today?",
        "Who owns the data platform budget?",
    )
    assert len(brief.discovery_questions) == 5
    assert brief.model_versions["llm"] == "scripted-llm"
    assert any("rejected 1 malformed" in w for w in brief.warnings)
    assert any("padded with rubric-based question" in w for w in brief.warnings)
    assert any("facts-only" in w for w in brief.warnings)
    assert any("used rule-based items" in w for w in brief.warnings)


def test_overview_emptied_by_validation_is_rebuilt(settings: AppSettings, qualified: Qualified) -> None:
    revenue = id_of(qualified, "$12.4 billion")
    reply = {
        "company_overview": [
            {
                "text": "Acme has 1 million stores on Mars.",
                "provenance": "verified_fact",
                "evidence_ids": [revenue],
            }
        ],
        "executive_summary": [{"text": "Summary.", "provenance": "ai_recommendation"}],
        "executive_talking_points": [],
        "recommended_next_actions": [],
    }
    llm = ScriptedLLM(routes={WRITER_KEY: reply, QUESTIONS_KEY: "garbage", OPPORTUNITY_KEY: "garbage"})
    brief = BriefGenerationAgent(llm, default_registry(), settings).generate(
        run_id="run-3", company=COMPANY, qualification=qualified.result, evidence=qualified.evidence
    )
    assert brief.company_overview
    assert all(s.provenance is FACT for s in brief.company_overview)
    assert any("rebuilt deterministically" in w for w in brief.warnings)
    assert any("discovery questions LLM failed" in w for w in brief.warnings)
    assert any("opportunity analysis LLM failed" in w for w in brief.warnings)
    assert any("LLM produced no usable statements" in w for w in brief.warnings)
    assert "Acme has 1 million stores on Mars." in brief.citation_report.removed_statements


def test_writer_failure_uses_deterministic_sections(settings: AppSettings, qualified: Qualified) -> None:
    def broken(messages: Sequence[ChatMessage]) -> str:
        return "not json"

    llm = ScriptedLLM(
        routes={WRITER_KEY: broken, OPPORTUNITY_KEY: {}, QUESTIONS_KEY: questions_reply(["Q?"] * 5)}
    )
    brief = BriefGenerationAgent(llm, default_registry(), settings).generate(
        run_id="run-4", company=COMPANY, qualification=qualified.result, evidence=qualified.evidence
    )
    assert any("brief writer LLM failed" in w for w in brief.warnings)
    assert brief.recommended_next_actions
    assert brief.discovery_questions[0] == "Q?"
    assert len(set(brief.discovery_questions)) == 5


def test_no_evidence_brief(settings: AppSettings) -> None:
    result = ScoringEngine(settings.scoring).evaluate([])
    brief = BriefGenerationAgent(ScriptedLLM(default="{}"), default_registry(), settings).generate(
        run_id="run-5", company=COMPANY, qualification=result, evidence=EvidenceRegistry()
    )
    assert brief.qualification.verdict is FitVerdict.NOT_ENOUGH_EVIDENCE
    assert brief.company_overview == ()
    assert brief.evidence == ()
    assert brief.executive_summary
    assert len(brief.discovery_questions) == 5


def test_builder_next_actions_per_verdict() -> None:
    builder = DeterministicBriefBuilder()
    engine = ScoringEngine(ScoringSettings())
    good = builder.next_actions(COMPANY, engine.evaluate(uniform_scores(5)))
    potential = builder.next_actions(COMPANY, engine.evaluate(uniform_scores(3)))
    weak = builder.next_actions(COMPANY, engine.evaluate(uniform_scores(1)))
    assert "executive sponsor" in good[0].text
    assert "Validate the evidence gap" in potential[0].text
    assert "Gather more public evidence" in weak[0].text
    assert builder.section("unknown", COMPANY, engine.evaluate([]), EvidenceRegistry()) == []
    empty = engine.evaluate([])
    assert "Open by asking" in builder.talking_points(COMPANY, empty, EvidenceRegistry())[0].text


def test_merge_reports() -> None:
    check = CitationCheck(evidence_id="E1", url="https://a", supported=True, support_score=1.0, reason="ok")
    merged = merge_reports(
        CitationReport(
            checks=(check,), total_statements=2, supported_statements=1, removed_statements=("x",)
        ),
        CitationReport(total_statements=1, supported_statements=1),
    )
    assert (merged.total_statements, merged.supported_statements, len(merged.checks)) == (3, 2, 1)
    assert merged.removed_statements == ("x",)
