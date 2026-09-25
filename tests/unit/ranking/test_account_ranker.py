from __future__ import annotations

import pytest

from client_research_agent.config.settings import ScoringSettings
from client_research_agent.models import Criterion, FitVerdict, QualificationResult
from client_research_agent.ranking.account_ranker import AccountRanker
from client_research_agent.scoring.engine import ScoringEngine
from tests.unit.qualification.helpers import make_score, uniform_scores


def result(score: int, confidence: float, near_term: int | None = None) -> QualificationResult:
    scores = uniform_scores(score, confidence)
    if near_term is not None:
        scores[-1] = make_score(Criterion.NEAR_TERM_OPPORTUNITY, near_term, confidence, ("E9",))
    return ScoringEngine(ScoringSettings()).evaluate(scores)


def test_tier_then_lower_bound_ordering() -> None:
    speculative = result(4, 0.5)
    solid = result(4, 0.95)
    potential = result(3, 0.95)
    thin = result(5, 0.2)
    ranked = AccountRanker().rank(
        [("Speculative", speculative), ("Thin", thin), ("Potential", potential), ("Solid", solid)]
    )
    assert [r.company for r in ranked] == ["Solid", "Speculative", "Potential", "Thin"]
    assert [r.rank for r in ranked] == [1, 2, 3, 4]
    assert ranked[0].verdict is FitVerdict.GOOD_FIT
    assert ranked[-1].verdict is FitVerdict.NOT_ENOUGH_EVIDENCE
    assert ranked[0].lower_bound == pytest.approx(4.0 - 0.05 * 2.0)


def test_tie_breaks_on_near_term_then_name() -> None:
    base = result(4, 0.9, near_term=4)
    same = result(4, 0.9, near_term=4)
    timely = result(4, 0.9, near_term=5)
    ranked = AccountRanker(uncertainty_span=0.0).rank([("Beta", base), ("alpha", same), ("Gamma", timely)])
    assert [r.company for r in ranked] == ["Gamma", "alpha", "Beta"]
    assert ranked[0].near_term_score == 5
    row = ranked[0].as_row()
    assert row["company"] == "Gamma"
    assert row["verdict"] == "good_fit"


def test_duplicates_and_validation() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        AccountRanker().rank([("Acme", result(3, 0.8)), ("ACME", result(3, 0.8))])
    with pytest.raises(ValueError, match="uncertainty_span"):
        AccountRanker(uncertainty_span=-1)
    assert AccountRanker().rank([]) == []


def test_missing_near_term_score_defaults_to_zero() -> None:
    partial = QualificationResult(
        scores=(make_score(Criterion.COMPANY_SCALE, 3),),
        weighted_score=0.6,
        overall_confidence=0.5,
        verdict=FitVerdict.NOT_ENOUGH_EVIDENCE,
        verdict_rationale="partial",
    )
    assert AccountRanker().rank([("Solo", partial)])[0].near_term_score == 0
