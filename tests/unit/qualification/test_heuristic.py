from __future__ import annotations

from datetime import date

from client_research_agent.config.settings import ScoringSettings
from client_research_agent.models import Criterion, DocumentType
from client_research_agent.qualification.criteria import get_definition
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.qualification.heuristic import (
    MAX_HEURISTIC_CONFIDENCE,
    NO_EVIDENCE_CONFIDENCE,
    HeuristicQualifier,
)
from tests.support.doubles import make_chunk
from tests.unit.qualification.helpers import TODAY, acme_corpus, registry_for


def _qualifier() -> HeuristicQualifier:
    return HeuristicQualifier(ScoringSettings(), today=TODAY)


def test_no_evidence_scores_zero_with_minimal_confidence() -> None:
    score = _qualifier().score(Criterion.AI_DATA_FOCUS, EvidenceRegistry(), [])
    assert score.score == 0
    assert score.confidence == NO_EVIDENCE_CONFIDENCE
    assert score.evidence_ids == ()
    assert "No public evidence" in score.rationale
    assert score.weight == 0.25


def test_strong_ai_evidence_scores_high_and_cites_contributors() -> None:
    corpus = acme_corpus()
    registry = registry_for(corpus)
    score = _qualifier().score(Criterion.AI_DATA_FOCUS, registry, registry.ids())
    assert score.score >= 3
    assert "E3" in score.evidence_ids
    assert score.evidence_ids[0] == "E3"
    assert 0.3 < score.confidence <= MAX_HEURISTIC_CONFIDENCE
    assert any("E3" in line for line in score.reasoning_trace)
    assert "Supporting signals" in score.rationale


def test_negative_signals_reduce_score() -> None:
    positive = make_chunk(
        "p", "Acme appoints a new chief data officer and announces a launch next fiscal year.", doc_id="p"
    )
    negative = make_chunk(
        "n", "Acme announced layoffs, a hiring freeze and a budget freeze during restructuring.", doc_id="n"
    )
    qualifier = _qualifier()
    good = qualifier.score(Criterion.NEAR_TERM_OPPORTUNITY, registry_for([positive]), ["E1"])
    mixed_registry = registry_for([positive, negative])
    mixed = qualifier.score(Criterion.NEAR_TERM_OPPORTUNITY, mixed_registry, ["E1", "E2"])
    assert mixed.score < good.score
    assert "Countervailing signals" in mixed.rationale


def test_scale_uses_numeric_bands() -> None:
    chunk = make_chunk(
        "s",
        "Acme reported annual revenue of $12.4 billion. It has 45,000 employees.",
        document_type=DocumentType.SEC_FILING,
    )
    score = _qualifier().score(Criterion.COMPANY_SCALE, registry_for([chunk]), ["E1"])
    assert score.score == 5
    assert score.evidence_ids == ("E1",)
    assert "Revenue/headcount figures" in score.rationale
    assert any("band 5" in line for line in score.reasoning_trace)


def test_small_company_numeric_band_is_low() -> None:
    chunk = make_chunk("s", "The startup has 40 employees and a seed round behind it.")
    score = _qualifier().score(Criterion.COMPANY_SCALE, registry_for([chunk]), ["E1"])
    assert score.score == 1


def test_scale_without_figures_is_capped() -> None:
    chunk = make_chunk(
        "g",
        "A global multinational listed on NYSE and NASDAQ with subsidiaries in many countries, "
        "worldwide segments and a headquarters; a Fortune 500 publicly traded market capitalization leader.",
    )
    registry = registry_for([chunk, make_chunk("g2", chunk.text + " Again global.", doc_id="d2")])
    score = _qualifier().score(Criterion.COMPANY_SCALE, registry, registry.ids())
    assert score.score == 3
    assert any("capped at 3" in line for line in score.reasoning_trace)


def test_stale_evidence_counts_less_than_fresh() -> None:
    text = "Acme is investing in machine learning, generative AI and a lakehouse data platform."
    fresh = registry_for([make_chunk("f", text, publication_date=date(2026, 8, 1))])
    stale = registry_for([make_chunk("s", text, publication_date=date(2016, 1, 1))])
    qualifier = _qualifier()
    fresh_score = qualifier.score(Criterion.AI_DATA_FOCUS, fresh, ["E1"])
    stale_score = qualifier.score(Criterion.AI_DATA_FOCUS, stale, ["E1"])
    assert fresh_score.score > stale_score.score
    assert fresh_score.confidence > stale_score.confidence


def test_irrelevant_evidence_has_low_confidence_and_no_citations() -> None:
    registry = registry_for([make_chunk("x", "The weather was pleasant at the picnic.")])
    score = _qualifier().score(Criterion.TECH_MODERNIZATION, registry, ["E1", "E404"])
    assert score.score == 0
    assert score.evidence_ids == ()
    assert score.confidence <= 0.25
    assert "no recognised signals" in score.rationale


def test_signal_hits_expose_matches() -> None:
    registry = registry_for(acme_corpus())
    hits = _qualifier().signal_hits(get_definition(Criterion.TECH_MODERNIZATION), registry, ["E2"])
    assert hits[0].positive
    assert hits[0].contribution > 0


def test_default_today_is_used_when_not_pinned() -> None:
    assert HeuristicQualifier(ScoringSettings()).today == date.today()
