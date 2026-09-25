from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from client_research_agent.config.settings import AppSettings
from client_research_agent.models import Criterion
from client_research_agent.prompts.registry import default_registry
from client_research_agent.qualification.agent import QualificationAgent
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.services.ports import ChatMessage, LLMResponse
from client_research_agent.utils.errors import CircuitOpenError, UpstreamTimeoutError
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.qualification.helpers import COMPANY, TODAY, KeywordRetriever, acme_corpus

_IDS = re.compile(r"Only these ids exist: ([E0-9, ]+)\.")


def shown_ids(messages: Sequence[ChatMessage]) -> list[str]:
    match = _IDS.search("\n".join(m.content for m in messages))
    return [i.strip() for i in match.group(1).split(",")] if match else []


def responder(
    score: int, confidence: float = 0.9, extra_ids: Sequence[str] = (), cite: bool = True
) -> Callable[[Sequence[ChatMessage]], Mapping[str, Any]]:
    def reply(messages: Sequence[ChatMessage]) -> Mapping[str, Any]:
        ids = shown_ids(messages)[:2] if cite else []
        return {
            "score": score,
            "confidence": confidence,
            "rationale": f"Evidence supports level {score}.",
            "evidence_ids": [*ids, *extra_ids],
            "reasoning_steps": [f"Reviewed {', '.join(ids) or 'nothing'}"],
        }

    return reply


def title_key(title: str) -> str:
    return f"ONE criterion: **{title}**"


@dataclass
class RaisingLLM:
    error: Exception

    @property
    def model_name(self) -> str:
        return "raising-llm"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        raise self.error


def make_agent(
    settings: AppSettings, llm: Any, retriever: KeywordRetriever, **kwargs: Any
) -> QualificationAgent:
    return QualificationAgent(llm, retriever, default_registry(), settings, today=TODAY, **kwargs)


def test_heuristic_only_when_no_llm(settings: AppSettings) -> None:
    retriever = KeywordRetriever(acme_corpus())
    output = make_agent(settings, None, retriever).qualify(COMPANY)
    assert [s.criterion for s in output.scores] == list(Criterion)
    assert output.llm_criteria == frozenset()
    assert output.score_for(Criterion.COMPANY_SCALE).score == 5
    assert all(s.evidence_ids for s in output.scores)
    assert len(output.evidence) == 5
    assert output.warnings == ()
    assert set(output.heuristic_scores) == set(Criterion)
    assert all(ids for ids in output.evidence_by_criterion.values())
    assert len(retriever.queries) == 15
    assert all(COMPANY in q for q in retriever.queries)


def test_evidence_ids_are_deterministic_across_runs(settings: AppSettings) -> None:
    first = make_agent(settings, None, KeywordRetriever(acme_corpus())).qualify(COMPANY)
    second = make_agent(settings, None, KeywordRetriever(acme_corpus()), max_workers=1).qualify(COMPANY)
    assert [(e.evidence_id, e.chunk_id) for e in first.evidence] == [
        (e.evidence_id, e.chunk_id) for e in second.evidence
    ]
    assert first.scores == second.scores


def test_llm_scores_are_used_when_grounded(settings: AppSettings) -> None:
    llm = ScriptedLLM(
        routes={
            title_key("Company Size & Scale"): responder(5, 0.9),
            title_key("Technology Modernization"): responder(4, 0.8),
            title_key("AI and Data Focus"): responder(4, 0.85),
            title_key("Industry Trends"): responder(3, 0.7),
            title_key("Near-Term Opportunity"): responder(4, 0.8),
        }
    )
    output = make_agent(settings, llm, KeywordRetriever(acme_corpus())).qualify(COMPANY)
    assert output.llm_criteria == frozenset(Criterion)
    scale = output.score_for(Criterion.COMPANY_SCALE)
    assert scale.score == 5
    assert scale.confidence == pytest.approx(0.9)
    assert scale.rationale == "Evidence supports level 5."
    assert any(line.startswith("heuristic cross-check") for line in scale.reasoning_trace)
    assert len(llm.calls) == 5
    prompt = "\n".join(m.content for m in llm.calls[0])
    assert "<evidence>" in prompt
    assert "</evidence>" in prompt


def test_disagreement_with_heuristic_lowers_confidence(settings: AppSettings) -> None:
    llm = ScriptedLLM(default={}, routes={title_key("Company Size & Scale"): responder(1, 0.9)})
    output = make_agent(settings, llm, KeywordRetriever(acme_corpus())).qualify(COMPANY)
    scale = output.score_for(Criterion.COMPANY_SCALE)
    assert output.heuristic_scores[Criterion.COMPANY_SCALE].score == 5
    assert scale.score == 1
    assert scale.confidence == pytest.approx(0.9 * (1 - 0.15 * 3))
    assert any("disagree strongly" in w for w in output.warnings)


def test_unknown_ids_are_dropped(settings: AppSettings) -> None:
    llm = ScriptedLLM(routes={title_key("AI and Data Focus"): responder(4, 0.8, extra_ids=("E99",))})
    output = make_agent(settings, llm, KeywordRetriever(acme_corpus())).qualify(COMPANY)
    ai = output.score_for(Criterion.AI_DATA_FOCUS)
    assert "E99" not in ai.evidence_ids
    assert ai.evidence_ids
    assert Criterion.AI_DATA_FOCUS in output.llm_criteria
    assert any("dropped unknown evidence ids" in w and "E99" in w for w in output.warnings)


def test_ungrounded_llm_answer_falls_back_to_heuristic(settings: AppSettings) -> None:
    llm = ScriptedLLM(
        routes={title_key("Industry Trends"): responder(5, 0.95, extra_ids=("E42",), cite=False)}
    )
    output = make_agent(settings, llm, KeywordRetriever(acme_corpus())).qualify(COMPANY)
    trends = output.score_for(Criterion.INDUSTRY_TRENDS)
    heuristic = output.heuristic_scores[Criterion.INDUSTRY_TRENDS]
    assert trends.score == heuristic.score
    assert trends.confidence <= 0.3
    assert "LLM assessment discarded" in trends.rationale
    assert Criterion.INDUSTRY_TRENDS not in output.llm_criteria
    assert any("not grounded" in w for w in output.warnings)


def test_single_citation_caps_confidence_when_minimum_is_higher(settings: AppSettings) -> None:
    tuned = settings.model_copy(
        update={"scoring": settings.scoring.model_copy(update={"min_evidence_per_criterion": 3})}
    )

    def one_id(messages: Sequence[ChatMessage]) -> Mapping[str, Any]:
        reply = dict(responder(5, 0.95)(messages))
        reply["evidence_ids"] = shown_ids(messages)[:1]
        return reply

    llm = ScriptedLLM(routes={title_key("Company Size & Scale"): one_id})
    output = make_agent(tuned, llm, KeywordRetriever(acme_corpus())).qualify(COMPANY)
    assert output.score_for(Criterion.COMPANY_SCALE).confidence <= 0.4


@pytest.mark.parametrize(
    "llm",
    [
        ScriptedLLM(default="this is not json at all"),
        RaisingLLM(UpstreamTimeoutError("serving endpoint timed out")),
        RaisingLLM(CircuitOpenError("chat", 12.0)),
    ],
)
def test_llm_failures_degrade_to_heuristic(settings: AppSettings, llm: Any) -> None:
    output = make_agent(settings, llm, KeywordRetriever(acme_corpus())).qualify(COMPANY)
    assert output.llm_criteria == frozenset()
    for score in output.scores:
        heuristic = output.heuristic_scores[score.criterion]
        assert score.score == heuristic.score
        assert score.reasoning_trace[-1].startswith("LLM unavailable")
    assert sum("used deterministic heuristic score" in w for w in output.warnings) == 5


def test_retrieval_failures_and_foreign_chunks_are_reported(settings: AppSettings) -> None:
    corpus = [*acme_corpus(), make_chunk("other", "Globex annual revenue of $9 billion.", company="Globex")]
    retriever = KeywordRetriever(corpus, fail_on=("headcount",), ignore_company=True)
    output = make_agent(settings, None, retriever).qualify(COMPANY)
    assert any("retrieval failed" in w for w in output.warnings)
    assert any("other companies" in w for w in output.warnings)
    assert all(e.chunk_id != "other" for e in output.evidence)


def test_empty_corpus_yields_zero_scores_and_warnings(settings: AppSettings) -> None:
    llm = ScriptedLLM()
    output = make_agent(settings, llm, KeywordRetriever([])).qualify(COMPANY)
    assert all(s.score == 0 for s in output.scores)
    assert len(output.evidence) == 0
    assert llm.calls == []
    assert sum("no public evidence retrieved" in w for w in output.warnings) == 5


def test_existing_registry_is_extended(settings: AppSettings) -> None:
    registry = EvidenceRegistry()
    output = make_agent(settings, None, KeywordRetriever(acme_corpus())).qualify(COMPANY, evidence=registry)
    assert output.evidence is registry
    assert len(registry) == 5


def test_score_for_unknown_criterion_raises(settings: AppSettings) -> None:
    output = make_agent(settings, None, KeywordRetriever([])).qualify(COMPANY)
    trimmed = type(output)(scores=output.scores[:1], evidence=output.evidence, warnings=())
    with pytest.raises(KeyError):
        trimmed.score_for(Criterion.NEAR_TERM_OPPORTUNITY)


def test_prompt_versions_are_exposed(settings: AppSettings) -> None:
    agent = make_agent(settings, None, KeywordRetriever([]))
    assert list(agent.prompt_versions) == ["prompt:criterion_qualifier"]
