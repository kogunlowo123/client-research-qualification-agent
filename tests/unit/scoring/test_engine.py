from __future__ import annotations

import pytest

from client_research_agent.config.settings import ScoringSettings
from client_research_agent.models import Criterion, FitVerdict
from client_research_agent.scoring.engine import ScoringEngine, VerdictReason
from tests.unit.qualification.helpers import make_score, uniform_scores


@pytest.fixture
def engine() -> ScoringEngine:
    return ScoringEngine(ScoringSettings())


def test_good_fit(engine: ScoringEngine) -> None:
    breakdown = engine.breakdown(uniform_scores(4, 0.8))
    result = breakdown.result
    assert result.verdict is FitVerdict.GOOD_FIT
    assert breakdown.reason is VerdictReason.GOOD_FIT
    assert result.weighted_score == pytest.approx(4.0)
    assert result.overall_confidence == pytest.approx(0.8)
    assert result.verdict_rationale.startswith("Good fit")
    assert sum(breakdown.contributions.values()) == pytest.approx(4.0)
    assert breakdown.under_evidenced == ()


def test_weighted_score_uses_configured_weights(engine: ScoringEngine) -> None:
    scores = [
        make_score(Criterion.COMPANY_SCALE, 5),
        make_score(Criterion.TECH_MODERNIZATION, 0),
        make_score(Criterion.AI_DATA_FOCUS, 4),
        make_score(Criterion.INDUSTRY_TRENDS, 2),
        make_score(Criterion.NEAR_TERM_OPPORTUNITY, 3),
    ]
    result = engine.evaluate(scores)
    assert result.weighted_score == pytest.approx(5 * 0.2 + 0 + 4 * 0.25 + 2 * 0.1 + 3 * 0.25)
    assert result.verdict is FitVerdict.POTENTIAL_FIT
    assert "Potential fit" in result.verdict_rationale


def test_stale_weights_are_replaced(engine: ScoringEngine) -> None:
    scores = uniform_scores(3)
    scores[0] = scores[0].model_copy(update={"weight": 0.9})
    result = engine.evaluate(scores)
    assert result.scores[0].weight == 0.2
    assert result.weighted_score == pytest.approx(3.0)


def test_low_but_well_evidenced_score_is_not_enough_evidence_of_fit(engine: ScoringEngine) -> None:
    breakdown = engine.breakdown(uniform_scores(1, 0.9))
    assert breakdown.result.verdict is FitVerdict.NOT_ENOUGH_EVIDENCE
    assert breakdown.reason is VerdictReason.LOW_FIT
    assert "Evidence indicates low fit" in breakdown.result.verdict_rationale
    assert "insufficient evidence of fit" in breakdown.result.verdict_rationale


def test_two_criteria_without_evidence_is_not_enough_evidence(engine: ScoringEngine) -> None:
    scores = uniform_scores(5, 0.95)
    scores[1] = scores[1].model_copy(update={"evidence_ids": ()})
    scores[3] = scores[3].model_copy(update={"evidence_ids": ()})
    breakdown = engine.breakdown(scores)
    assert breakdown.result.verdict is FitVerdict.NOT_ENOUGH_EVIDENCE
    assert breakdown.reason is VerdictReason.MISSING_EVIDENCE
    assert breakdown.without_evidence == (Criterion.TECH_MODERNIZATION, Criterion.INDUSTRY_TRENDS)
    assert "Technology Modernization" in breakdown.result.verdict_rationale


def test_one_missing_criterion_penalises_confidence(engine: ScoringEngine) -> None:
    scores = uniform_scores(4, 0.8)
    scores[4] = scores[4].model_copy(update={"evidence_ids": ()})
    result = engine.evaluate(scores)
    expected = (0.75 * 0.8 + 0.25 * 0.8 * 0.25) * 0.9
    assert result.overall_confidence == pytest.approx(expected)
    assert result.verdict is FitVerdict.GOOD_FIT


def test_low_confidence_is_not_enough_evidence(engine: ScoringEngine) -> None:
    breakdown = engine.breakdown(uniform_scores(5, 0.3))
    assert breakdown.result.verdict is FitVerdict.NOT_ENOUGH_EVIDENCE
    assert breakdown.reason is VerdictReason.LOW_CONFIDENCE
    assert "below the 45% minimum" in breakdown.result.verdict_rationale


def test_missing_criteria_are_filled_with_zero(engine: ScoringEngine) -> None:
    result = engine.evaluate([make_score(Criterion.AI_DATA_FOCUS, 5)])
    assert [s.criterion for s in result.scores] == list(Criterion)
    filled = result.scores[0]
    assert filled.score == 0
    assert filled.confidence == 0.0
    assert "No assessment" in filled.rationale
    assert result.verdict is FitVerdict.NOT_ENOUGH_EVIDENCE


def test_duplicate_scores_are_rejected(engine: ScoringEngine) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        engine.evaluate([make_score(Criterion.AI_DATA_FOCUS, 5), make_score(Criterion.AI_DATA_FOCUS, 4)])


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="potential_fit_threshold"):
        ScoringEngine(ScoringSettings(good_fit_threshold=2.0, potential_fit_threshold=3.0))


def test_min_evidence_zero_disables_under_evidence_penalty() -> None:
    engine = ScoringEngine(ScoringSettings(min_evidence_per_criterion=0))
    scores = uniform_scores(4, 0.8)
    scores[0] = scores[0].model_copy(update={"evidence_ids": ()})
    breakdown = engine.breakdown(scores)
    assert breakdown.result.overall_confidence == pytest.approx(0.8)
    assert breakdown.under_evidenced == ()
    assert breakdown.without_evidence == (Criterion.COMPANY_SCALE,)
    assert engine.settings.min_evidence_per_criterion == 0


def test_sensitivity_detects_flips(engine: ScoringEngine) -> None:
    result = engine.evaluate([*uniform_scores(3, 0.8)[:4], make_score(Criterion.NEAR_TERM_OPPORTUNITY, 5)])
    assert result.weighted_score == pytest.approx(3.5)
    report = engine.sensitivity(result)
    assert report.base_verdict is FitVerdict.GOOD_FIT
    assert not report.robust
    assert all(entry.delta == -1 for entry in report.flips)
    assert {e.verdict for e in report.flips} == {FitVerdict.POTENTIAL_FIT}
    near_term = [e for e in report.entries if e.criterion is Criterion.NEAR_TERM_OPPORTUNITY]
    assert [e.delta for e in near_term] == [-1]
    assert "sensitive" in report.summary()


def test_sensitivity_robust_case(engine: ScoringEngine) -> None:
    report = engine.sensitivity(uniform_scores(5, 0.9))
    assert report.robust
    assert len(report.entries) == 5
    assert "robust" in report.summary()
    assert report.base_weighted_score == pytest.approx(5.0)
