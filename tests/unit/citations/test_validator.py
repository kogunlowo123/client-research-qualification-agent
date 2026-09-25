from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from client_research_agent.citations.validator import (
    PARTIAL_SUPPORT_NOTE,
    CitationValidator,
    is_http_url,
)
from client_research_agent.config.settings import GuardrailSettings
from client_research_agent.models import BriefStatement, ProvenanceKind
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.services.ports import ChatMessage, LLMResponse
from client_research_agent.utils.errors import UpstreamServiceError
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.qualification.helpers import acme_corpus, registry_for

FACT = ProvenanceKind.VERIFIED_FACT
REC = ProvenanceKind.AI_RECOMMENDATION


def fact(text: str, *ids: str) -> BriefStatement:
    return BriefStatement(text=text, provenance=FACT, evidence_ids=ids)


def rec(text: str, *ids: str) -> BriefStatement:
    return BriefStatement(text=text, provenance=REC, evidence_ids=ids)


@dataclass
class FailingJudge:
    calls: int = 0

    @property
    def model_name(self) -> str:
        return "failing-judge"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        raise UpstreamServiceError("judge endpoint returned 503", status_code=503)


@pytest.fixture
def registry() -> EvidenceRegistry:
    return registry_for(acme_corpus())


def test_supported_fact_is_kept_with_supporting_ids_only(registry: EvidenceRegistry) -> None:
    validator = CitationValidator(None, GuardrailSettings())
    outcome = validator.validate(
        {
            "overview": [
                fact("Acme Corp reported annual revenue of $12.4 billion for fiscal 2025.", "E1", "E3")
            ]
        },
        registry,
    )
    kept = outcome.sections["overview"]
    assert kept == (fact("Acme Corp reported annual revenue of $12.4 billion for fiscal 2025.", "E1"),)
    report = outcome.report
    assert (report.total_statements, report.supported_statements) == (1, 1)
    assert report.coverage == 1.0
    assert {c.evidence_id: c.supported for c in report.checks} == {"E1": True, "E3": False}
    assert all(c.url.startswith("https://") for c in report.checks)
    assert outcome.warnings == ()


def test_fabricated_figure_is_removed_not_downgraded(registry: EvidenceRegistry) -> None:
    statement = fact("Acme Corp reported annual revenue of $19.9 billion for fiscal 2025.", "E1")
    outcome = CitationValidator(None, GuardrailSettings()).validate({"overview": [statement]}, registry)
    assert outcome.sections["overview"] == ()
    assert outcome.report.removed_statements == (statement.text,)
    assert outcome.report.coverage == 0.0
    assert "figures not in evidence" in outcome.report.checks[0].reason
    assert any("removed 1 statement" in w for w in outcome.warnings)


def test_partial_support_is_downgraded_to_recommendation(registry: EvidenceRegistry) -> None:
    statement = fact(
        "Acme cloud migration implies vendor consolidation, outsourcing appetite, budget expansion and "
        "procurement reform.",
        "E2",
    )
    outcome = CitationValidator(None, GuardrailSettings()).validate({"priorities": [statement]}, registry)
    (result,) = outcome.sections["priorities"]
    assert result.provenance is REC
    assert result.text.endswith(PARTIAL_SUPPORT_NOTE)
    assert result.evidence_ids == ("E2",)
    assert outcome.downgraded == (statement.text,)
    assert outcome.report.supported_statements == 0
    assert any("re-labelled 1" in w for w in outcome.warnings)


def test_unsupported_fact_and_unknown_ids_are_removed(registry: EvidenceRegistry) -> None:
    unrelated = fact("The company won a national football championship.", "E1")
    unknown = fact("Acme Corp reported annual revenue of $12.4 billion.", "E77")
    outcome = CitationValidator(None, GuardrailSettings()).validate({"s": [unrelated, unknown]}, registry)
    assert outcome.sections["s"] == ()
    assert len(outcome.report.removed_statements) == 2
    unknown_check = next(c for c in outcome.report.checks if c.evidence_id == "E77")
    assert unknown_check.reason == "unknown evidence id"
    assert unknown_check.url == ""
    assert any("invalid evidence" in w for w in outcome.warnings)


def test_inline_markers_are_used_and_stripped(registry: EvidenceRegistry) -> None:
    statement = fact("A chief data officer was appointed in March 2026 [E3].", "E3")
    outcome = CitationValidator(None, GuardrailSettings()).validate({"s": [statement]}, registry)
    (kept,) = outcome.sections["s"]
    assert kept.text == "A chief data officer was appointed in March 2026."
    assert kept.evidence_ids == ("E3",)
    marker_only = BriefStatement(text="[E3]", provenance=FACT, evidence_ids=("E3",))
    assert (
        CitationValidator(None, GuardrailSettings()).validate({"s": [marker_only]}, registry).sections["s"]
        == ()
    )


def test_recommendations_keep_text_and_lose_invalid_ids(registry: EvidenceRegistry) -> None:
    statement = rec("Lead with the lakehouse programme [E9].", "E3", "E42")
    outcome = CitationValidator(None, GuardrailSettings()).validate({"s": [statement]}, registry)
    (kept,) = outcome.sections["s"]
    assert kept.provenance is REC
    assert kept.evidence_ids == ("E3",)
    assert kept.text == "Lead with the lakehouse programme."
    assert outcome.report.total_statements == 0
    assert any("stripped invalid evidence ids" in w for w in outcome.warnings)


def test_evidence_urls_must_be_http_and_retrieved() -> None:
    bad = make_chunk("ftp", "Acme revenue grew.", url="ftp://files.example.com/acme")
    good = make_chunk("ok", "Acme revenue grew.", doc_id="ok")
    registry = registry_for([bad, good])
    validator = CitationValidator(None, GuardrailSettings())
    assert validator.evidence_problem("E1", registry) == "evidence URL is not http(s)"
    assert validator.evidence_problem("E2", registry) is None
    assert validator.evidence_problem("E2", registry, retrieved_urls={"https://other"}) == (
        "evidence URL is not among retrieved documents"
    )
    outcome = validator.validate(
        {"s": [fact("Acme revenue grew.", "E2")]}, registry, retrieved_urls={"https://x"}
    )
    assert outcome.sections["s"] == ()
    assert is_http_url("https://a.example/x")
    assert not is_http_url("javascript:alert(1)")


def test_url_mismatch_is_detected(registry: EvidenceRegistry) -> None:
    evidence = registry.require("E1")
    registry._evidence["E1"] = evidence.model_copy(update={"url": "https://attacker.example/x"})
    problem = CitationValidator(None, GuardrailSettings()).evidence_problem("E1", registry)
    assert problem == "evidence URL does not match the retrieved document"


def test_judge_can_lower_and_modestly_raise_support(registry: EvidenceRegistry) -> None:
    statement = fact(
        "Acme cloud migration implies vendor consolidation, outsourcing appetite, budget expansion and "
        "procurement reform.",
        "E2",
    )
    doubting = ScriptedLLM(default={"supported": False, "score": 0.0, "reason": "not stated"})
    outcome = CitationValidator(doubting, GuardrailSettings()).validate({"s": [statement]}, registry)
    assert outcome.sections["s"] == ()
    assert "judge: not stated" in outcome.report.checks[0].reason
    prompt = "\n".join(m.content for m in doubting.calls[0])
    assert "<statement>" in prompt
    assert "<evidence>" in prompt

    believing = ScriptedLLM(default={"supported": True, "score": 1.0, "reason": ""})
    outcome = CitationValidator(believing, GuardrailSettings()).validate({"s": [statement]}, registry)
    (kept,) = outcome.sections["s"]
    assert kept.provenance is FACT
    check = outcome.report.checks[0]
    assert check.support_score <= 1.0
    assert "judge: supported" in check.reason


def test_judge_is_not_consulted_for_number_mismatch(registry: EvidenceRegistry) -> None:
    judge = ScriptedLLM(default={"supported": True, "score": 1.0, "reason": "ok"})
    statement = fact("Acme Corp reported annual revenue of $99 billion for fiscal 2025.", "E1")
    outcome = CitationValidator(judge, GuardrailSettings()).validate({"s": [statement]}, registry)
    assert outcome.sections["s"] == ()
    assert judge.calls == []


def test_judge_failure_falls_back_to_lexical_once(registry: EvidenceRegistry) -> None:
    judge = FailingJudge()
    statements = [
        fact("Acme Corp reported annual revenue of $12.4 billion for fiscal 2025.", "E1"),
        fact("A chief data officer was appointed in March 2026.", "E3"),
    ]
    outcome = CitationValidator(judge, GuardrailSettings()).validate({"s": statements}, registry)
    assert len(outcome.sections["s"]) == 2
    assert judge.calls == 1
    assert sum("citation judge unavailable" in w for w in outcome.warnings) == 1


def test_judge_disabled_by_weight_or_budget(registry: EvidenceRegistry) -> None:
    judge = ScriptedLLM(default={"supported": True, "score": 1.0, "reason": "ok"})
    statement = fact("A chief data officer was appointed in March 2026.", "E3")
    CitationValidator(judge, GuardrailSettings(), judge_weight=0.0).validate({"s": [statement]}, registry)
    CitationValidator(judge, GuardrailSettings(), max_judge_calls=0).validate({"s": [statement]}, registry)
    assert judge.calls == []


def test_constructor_validation_and_properties() -> None:
    with pytest.raises(ValueError, match="judge_weight"):
        CitationValidator(None, GuardrailSettings(), judge_weight=2.0)
    with pytest.raises(ValueError, match="partial_support_ratio"):
        CitationValidator(None, GuardrailSettings(), partial_support_ratio=0.0)
    validator = CitationValidator(None, GuardrailSettings(min_citation_support=0.4))
    assert validator.threshold == 0.4
    assert list(validator.prompt_versions) == ["prompt:citation_judge"]
