"""Weighted scoring, verdict policy and sensitivity analysis.

Weighted score
    ``sum(score_c * weight_c)`` over the five criteria using the configured
    weights (which sum to 1.0), so the result stays on the 0-5 rubric scale.
    Weights from ``ScoringSettings`` are authoritative; a ``CriterionScore``
    carrying a stale weight is re-weighted. A criterion with no score at all is
    treated as score 0, confidence 0, no evidence.

Overall confidence
    The weight-averaged criterion confidence, where a criterion citing fewer
    than ``min_evidence_per_criterion`` evidence items contributes only a
    quarter of its stated confidence (an un-evidenced score is an opinion),
    and the average is further multiplied by ``1 - 0.1 * n_under_evidenced``.

Verdict policy (evaluated in order)
    1. NOT_ENOUGH_EVIDENCE when two or more criteria cite no evidence at all, or
       overall confidence is below ``min_confidence``. We cannot say anything
       reliable, so we say so.
    2. GOOD_FIT when the weighted score reaches ``good_fit_threshold``.
    3. POTENTIAL_FIT when it reaches ``potential_fit_threshold``.
    4. Otherwise NOT_ENOUGH_EVIDENCE with reason ``LOW_FIT``.

    Decision on case 4: the domain has no "poor fit" verdict. Reporting a
    well-evidenced low score as POTENTIAL_FIT would overstate the account and
    push it into the pipeline, which is the costly error for an account team.
    The honest reading of a low score is "the public evidence does not
    demonstrate fit", so the verdict is NOT_ENOUGH_EVIDENCE *of fit*, and the
    ``verdict_rationale`` says explicitly that the evidence indicates low fit
    (as opposed to missing evidence) so readers and the account ranker can tell
    the two apart. ``ScoreBreakdown.reason`` carries the machine-readable code.

Sensitivity
    For every criterion the score is moved by -1 and +1 (clamped to 0-5, with
    confidences and evidence held fixed) and the verdict recomputed, showing
    which single-criterion judgement calls could change the outcome.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from client_research_agent.config.settings import ScoringSettings
from client_research_agent.models import Criterion, CriterionScore, FitVerdict, QualificationResult
from client_research_agent.qualification.criteria import get_definition

UNDER_EVIDENCED_CONFIDENCE_FACTOR = 0.25
UNDER_EVIDENCED_GLOBAL_PENALTY = 0.1
MISSING_EVIDENCE_LIMIT = 2


class VerdictReason(StrEnum):
    GOOD_FIT = "good_fit"
    POTENTIAL_FIT = "potential_fit"
    LOW_FIT = "low_fit"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_EVIDENCE = "missing_evidence"


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    result: QualificationResult
    reason: VerdictReason
    contributions: Mapping[Criterion, float]
    under_evidenced: tuple[Criterion, ...]
    without_evidence: tuple[Criterion, ...]


@dataclass(frozen=True, slots=True)
class SensitivityEntry:
    criterion: Criterion
    delta: int
    score_before: int
    score_after: int
    weighted_score: float
    verdict: FitVerdict
    changes_verdict: bool


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    base_verdict: FitVerdict
    base_weighted_score: float
    entries: tuple[SensitivityEntry, ...]

    @property
    def flips(self) -> tuple[SensitivityEntry, ...]:
        return tuple(entry for entry in self.entries if entry.changes_verdict)

    @property
    def robust(self) -> bool:
        """True when no single +/-1 change to any criterion alters the verdict."""
        return not self.flips

    def summary(self) -> str:
        if self.robust:
            return f"Verdict {self.base_verdict.value} is robust to a +/-1 change in any single criterion."
        changes = "; ".join(
            f"{get_definition(e.criterion).title} {e.delta:+d} -> {e.verdict.value} ({e.weighted_score:.2f})"
            for e in self.flips
        )
        return f"Verdict {self.base_verdict.value} is sensitive: {changes}."


class ScoringEngine:
    def __init__(self, settings: ScoringSettings) -> None:
        if settings.potential_fit_threshold > settings.good_fit_threshold:
            raise ValueError("potential_fit_threshold must not exceed good_fit_threshold")
        self._settings = settings

    @property
    def settings(self) -> ScoringSettings:
        return self._settings

    # ------------------------------------------------------------------ public API
    def evaluate(self, scores: Iterable[CriterionScore]) -> QualificationResult:
        return self.breakdown(scores).result

    def breakdown(self, scores: Iterable[CriterionScore]) -> ScoreBreakdown:
        complete = self.normalize(scores)
        weighted = self._weighted(complete)
        confidence = self._overall_confidence(complete)
        under = tuple(s.criterion for s in complete if self._under_evidenced(s))
        missing = tuple(s.criterion for s in complete if not s.evidence_ids)
        verdict, reason = self._decide(weighted, confidence, len(missing))
        rationale = self._rationale(reason, weighted, confidence, complete, missing)
        result = QualificationResult(
            scores=complete,
            weighted_score=weighted,
            overall_confidence=confidence,
            verdict=verdict,
            verdict_rationale=rationale,
        )
        return ScoreBreakdown(
            result=result,
            reason=reason,
            contributions={s.criterion: round(s.weighted, 4) for s in complete},
            under_evidenced=under,
            without_evidence=missing,
        )

    def normalize(self, scores: Iterable[CriterionScore]) -> tuple[CriterionScore, ...]:
        """One score per criterion in canonical order, with configured weights applied."""
        by_criterion: dict[Criterion, CriterionScore] = {}
        for score in scores:
            if score.criterion in by_criterion:
                raise ValueError(f"duplicate score for criterion {score.criterion.value}")
            by_criterion[score.criterion] = score
        complete: list[CriterionScore] = []
        for criterion in Criterion:
            weight = self._settings.weights[criterion]
            existing = by_criterion.get(criterion)
            if existing is None:
                complete.append(
                    CriterionScore(
                        criterion=criterion,
                        score=0,
                        weight=weight,
                        confidence=0.0,
                        rationale=f"No assessment was produced for {get_definition(criterion).title}.",
                    )
                )
            elif abs(existing.weight - weight) > 1e-9:
                complete.append(existing.model_copy(update={"weight": weight}))
            else:
                complete.append(existing)
        return tuple(complete)

    def sensitivity(self, scores: Iterable[CriterionScore] | QualificationResult) -> SensitivityReport:
        source = scores.scores if isinstance(scores, QualificationResult) else scores
        complete = self.normalize(source)
        confidence = self._overall_confidence(complete)
        missing = sum(1 for s in complete if not s.evidence_ids)
        base_weighted = self._weighted(complete)
        base_verdict, _ = self._decide(base_weighted, confidence, missing)
        entries: list[SensitivityEntry] = []
        for index, score in enumerate(complete):
            for delta in (-1, 1):
                new_score = max(0, min(5, score.score + delta))
                if new_score == score.score:
                    continue
                variant = list(complete)
                variant[index] = score.model_copy(update={"score": new_score})
                weighted = self._weighted(variant)
                verdict, _ = self._decide(weighted, confidence, missing)
                entries.append(
                    SensitivityEntry(
                        criterion=score.criterion,
                        delta=delta,
                        score_before=score.score,
                        score_after=new_score,
                        weighted_score=weighted,
                        verdict=verdict,
                        changes_verdict=verdict is not base_verdict,
                    )
                )
        return SensitivityReport(
            base_verdict=base_verdict, base_weighted_score=base_weighted, entries=tuple(entries)
        )

    # ------------------------------------------------------------------ internals
    def _under_evidenced(self, score: CriterionScore) -> bool:
        return len(score.evidence_ids) < self._settings.min_evidence_per_criterion

    @staticmethod
    def _weighted(scores: Sequence[CriterionScore]) -> float:
        return round(max(0.0, min(5.0, sum(s.score * s.weight for s in scores))), 4)

    def _overall_confidence(self, scores: Sequence[CriterionScore]) -> float:
        total_weight = sum(s.weight for s in scores)
        if total_weight <= 0:
            return 0.0
        under = 0
        accumulated = 0.0
        for score in scores:
            effective = score.confidence
            if self._under_evidenced(score):
                under += 1
                effective *= UNDER_EVIDENCED_CONFIDENCE_FACTOR
            accumulated += score.weight * effective
        average = accumulated / total_weight
        return round(max(0.0, min(1.0, average * (1 - UNDER_EVIDENCED_GLOBAL_PENALTY * under))), 4)

    def _decide(self, weighted: float, confidence: float, missing: int) -> tuple[FitVerdict, VerdictReason]:
        if missing >= MISSING_EVIDENCE_LIMIT:
            return FitVerdict.NOT_ENOUGH_EVIDENCE, VerdictReason.MISSING_EVIDENCE
        if confidence < self._settings.min_confidence:
            return FitVerdict.NOT_ENOUGH_EVIDENCE, VerdictReason.LOW_CONFIDENCE
        if weighted >= self._settings.good_fit_threshold:
            return FitVerdict.GOOD_FIT, VerdictReason.GOOD_FIT
        if weighted >= self._settings.potential_fit_threshold:
            return FitVerdict.POTENTIAL_FIT, VerdictReason.POTENTIAL_FIT
        return FitVerdict.NOT_ENOUGH_EVIDENCE, VerdictReason.LOW_FIT

    def _rationale(
        self,
        reason: VerdictReason,
        weighted: float,
        confidence: float,
        scores: Sequence[CriterionScore],
        missing: Sequence[Criterion],
    ) -> str:
        s = self._settings
        headline = f"Weighted score {weighted:.2f}/5 at {confidence:.0%} overall confidence."
        ranked = sorted(scores, key=lambda x: (-x.weighted, x.criterion.value))
        strongest = get_definition(ranked[0].criterion).title
        weakest = get_definition(ranked[-1].criterion).title
        if reason is VerdictReason.MISSING_EVIDENCE:
            names = ", ".join(get_definition(c).title for c in missing)
            detail = (
                f"Not enough evidence: {len(missing)} criteria have no supporting public evidence ({names}); "
                "a fit verdict would be speculative."
            )
        elif reason is VerdictReason.LOW_CONFIDENCE:
            detail = (
                f"Not enough evidence: overall confidence is below the {s.min_confidence:.0%} "
                "minimum, so the "
                "score is not reliable enough to act on."
            )
        elif reason is VerdictReason.GOOD_FIT:
            detail = (
                f"Good fit: score meets the {s.good_fit_threshold:.2f} threshold; strongest criterion is "
                f"{strongest}, weakest is {weakest}."
            )
        elif reason is VerdictReason.POTENTIAL_FIT:
            detail = (
                f"Potential fit: score meets the {s.potential_fit_threshold:.2f} threshold but not the "
                f"{s.good_fit_threshold:.2f} good-fit threshold; strongest criterion is "
                f"{strongest}, weakest is "
                f"{weakest}."
            )
        else:
            detail = (
                "Evidence indicates low fit: the evidence is sufficient to assess, but the score "
                "is below the "
                f"{s.potential_fit_threshold:.2f} potential-fit threshold, so there is insufficient "
                "evidence of fit "
                f"(weakest criterion: {weakest})."
            )
        return f"{detail} {headline}"
