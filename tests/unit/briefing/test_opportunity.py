from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from client_research_agent.briefing.opportunity import (
    MAPPING_NOTE,
    TREND_THEMES,
    OpportunityAnalysisAgent,
    attributes_to_gartner,
    has_analyst_gartner_source,
    themes_catalogue,
)
from client_research_agent.config.settings import ScoringSettings
from client_research_agent.models import DocumentType, FitVerdict, ProvenanceKind
from client_research_agent.prompts.registry import default_registry
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.scoring.engine import ScoringEngine
from client_research_agent.services.ports import ChatMessage, LLMResponse
from client_research_agent.utils.errors import RateLimitedError
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.briefing.conftest import Qualified
from tests.unit.qualification.helpers import COMPANY, acme_corpus, registry_for, uniform_scores

REC = ProvenanceKind.AI_RECOMMENDATION


@dataclass
class ThrottledLLM:
    @property
    def model_name(self) -> str:
        return "throttled"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        raise RateLimitedError("429", retry_after_seconds=1.0)


def test_rule_based_analysis_is_complete(qualified: Qualified) -> None:
    analysis = OpportunityAnalysisAgent(None, default_registry()).analyze(
        COMPANY, qualified.result, qualified.evidence
    )
    assert not analysis.used_llm
    assert analysis.technology_priorities
    assert any(s.provenance is ProvenanceKind.VERIFIED_FACT for s in analysis.technology_priorities)
    assert analysis.gartner_insights
    for insight in analysis.gartner_insights:
        assert insight.provenance is REC
        assert MAPPING_NOTE in insight.text
        assert insight.evidence_ids
        assert not attributes_to_gartner(insight.text)
    assert analysis.opportunities
    assert analysis.risks
    assert set(analysis.sections()) == {
        "technology_priorities",
        "gartner_relevant_insights",
        "opportunities",
        "risks",
    }


def test_rules_surface_risk_sentences_and_weak_verdicts() -> None:
    chunk = make_chunk(
        "r", "Acme announced layoffs and a hiring freeze as part of restructuring. It sells tools."
    )
    registry = registry_for([chunk])
    result = ScoringEngine(ScoringSettings()).evaluate(uniform_scores(1, 0.3))
    analysis = OpportunityAnalysisAgent(None, default_registry()).analyze(COMPANY, result, registry)
    facts = [s for s in analysis.risks if s.provenance is ProvenanceKind.VERIFIED_FACT]
    assert facts[0].text == "Acme announced layoffs and a hiring freeze as part of restructuring."
    assert result.verdict is FitVerdict.NOT_ENOUGH_EVIDENCE
    assert any("Evidence gap" in s.text for s in analysis.risks)
    assert "No criterion" in analysis.opportunities[0].text


def test_empty_evidence_uses_rules() -> None:
    result = ScoringEngine(ScoringSettings()).evaluate([])
    llm = ScriptedLLM()
    analysis = OpportunityAnalysisAgent(llm, default_registry()).analyze(COMPANY, result, EvidenceRegistry())
    assert llm.calls == []
    assert analysis.technology_priorities == ()
    assert analysis.gartner_insights == ()
    assert any("exploratory" in s.text for s in analysis.risks)


def test_llm_analysis_is_filtered(qualified: Qualified) -> None:
    llm = ScriptedLLM(
        default={
            "technology_priorities": [
                {
                    "text": "Acme is investing in generative AI.",
                    "provenance": "verified_fact",
                    "evidence_ids": ["E3"],
                },
                {"text": "Uncited claim.", "provenance": "verified_fact", "evidence_ids": []},
            ],
            "gartner_insights": [
                {
                    "theme": "Data fabric",
                    "text": "The lakehouse aligns with the data fabric theme.",
                    "evidence_ids": ["E3"],
                },
                {
                    "theme": "Data fabric",
                    "text": "Gartner predicts Acme will lead data fabric.",
                    "evidence_ids": ["E3"],
                },
                {"theme": "Quantum supremacy", "text": "Unknown theme.", "evidence_ids": ["E3"]},
            ],
            "opportunities": [
                {"text": "Offer an AI acceleration engagement [E3].", "provenance": "ai_recommendation"}
            ],
            "risks": [{"text": "Execution risk on migration.", "evidence_ids": ["E2"]}],
        }
    )
    agent = OpportunityAnalysisAgent(llm, default_registry())
    analysis = agent.analyze(COMPANY, qualified.result, qualified.evidence)
    assert analysis.used_llm
    assert [s.text for s in analysis.technology_priorities] == ["Acme is investing in generative AI."]
    assert len(analysis.gartner_insights) == 1
    assert analysis.gartner_insights[0].text.startswith("Mapped to the 'Data fabric' trend theme:")
    assert analysis.opportunities[0].evidence_ids == ("E3",)
    assert any("attributing a claim to Gartner" in w for w in analysis.warnings)
    assert any("unknown theme" in w for w in analysis.warnings)
    assert any("dropped uncited fact" in w for w in analysis.warnings)
    prompt = "\n".join(m.content for m in llm.calls[0])
    assert themes_catalogue() in prompt
    assert list(agent.prompt_versions) == ["prompt:opportunity_analysis"]


def test_gartner_attribution_allowed_with_analyst_source() -> None:
    analyst = make_chunk(
        "g",
        "Gartner named data fabric a top strategic technology trend, citing Acme as an adopter.",
        document_type=DocumentType.ANALYST_PUBLIC,
    )
    registry = registry_for([analyst])
    assert has_analyst_gartner_source(registry, ["E1"])
    assert not has_analyst_gartner_source(registry_for(acme_corpus()), ["E1", "E99"])
    result = ScoringEngine(ScoringSettings()).evaluate(uniform_scores(4))
    llm = ScriptedLLM(
        default={
            "gartner_insights": [
                {
                    "theme": "data fabric",
                    "text": "Gartner named data fabric a top trend.",
                    "evidence_ids": ["E1"],
                }
            ]
        }
    )
    analysis = OpportunityAnalysisAgent(llm, default_registry()).analyze(COMPANY, result, registry)
    assert analysis.gartner_insights[0].evidence_ids == ("E1",)
    assert any("used rule-based items" in w for w in analysis.warnings)
    assert analysis.opportunities


def test_llm_failure_falls_back(qualified: Qualified) -> None:
    analysis = OpportunityAnalysisAgent(ThrottledLLM(), default_registry()).analyze(
        COMPANY, qualified.result, qualified.evidence
    )
    assert not analysis.used_llm
    assert analysis.opportunities
    assert any("RateLimitedError" in w for w in analysis.warnings)


def test_theme_catalogue_lists_every_theme() -> None:
    catalogue = themes_catalogue()
    assert all(theme.name in catalogue for theme in TREND_THEMES)
    assert attributes_to_gartner("According to Gartner, AI matters.")
    assert attributes_to_gartner("Gartner's latest research shows growth.")
    assert not attributes_to_gartner("This aligns with a Gartner trend theme.")
