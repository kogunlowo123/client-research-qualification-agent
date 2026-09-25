"""Portfolio prioritisation across several qualified accounts.

Accounts are ordered by a confidence-adjusted *lower bound* rather than the
raw weighted score, so a well-evidenced 3.6 outranks a speculative 3.9::

    lower_bound = max(0, weighted_score - (1 - overall_confidence) * uncertainty_span)

Ordering (all descending unless noted):

1. verdict tier - GOOD_FIT, then POTENTIAL_FIT, then NOT_ENOUGH_EVIDENCE, so an
   account can never jump a tier on score alone;
2. lower bound;
3. near-term opportunity score (timing breaks ties between similar accounts);
4. weighted score;
5. overall confidence;
6. company name ascending, for a fully deterministic order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from client_research_agent.models import Criterion, FitVerdict, QualificationResult

VERDICT_TIER: dict[FitVerdict, int] = {
    FitVerdict.GOOD_FIT: 0,
    FitVerdict.POTENTIAL_FIT: 1,
    FitVerdict.NOT_ENOUGH_EVIDENCE: 2,
}


@dataclass(frozen=True, slots=True)
class RankedAccount:
    rank: int
    company: str
    verdict: FitVerdict
    weighted_score: float
    overall_confidence: float
    lower_bound: float
    near_term_score: int

    def as_row(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "company": self.company,
            "verdict": self.verdict.value,
            "weighted_score": round(self.weighted_score, 3),
            "overall_confidence": round(self.overall_confidence, 3),
            "lower_bound": round(self.lower_bound, 3),
            "near_term_score": self.near_term_score,
        }


class AccountRanker:
    def __init__(self, *, uncertainty_span: float = 2.0) -> None:
        if uncertainty_span < 0:
            raise ValueError("uncertainty_span must be non-negative")
        self._span = uncertainty_span

    def lower_bound(self, result: QualificationResult) -> float:
        return max(0.0, result.weighted_score - (1.0 - result.overall_confidence) * self._span)

    @staticmethod
    def _near_term(result: QualificationResult) -> int:
        return next((s.score for s in result.scores if s.criterion is Criterion.NEAR_TERM_OPPORTUNITY), 0)

    def rank(self, accounts: Iterable[tuple[str, QualificationResult]]) -> list[RankedAccount]:
        entries = list(accounts)
        names = [name.casefold() for name, _ in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate company in account list")
        ordered = sorted(
            entries,
            key=lambda entry: (
                VERDICT_TIER[entry[1].verdict],
                -self.lower_bound(entry[1]),
                -self._near_term(entry[1]),
                -entry[1].weighted_score,
                -entry[1].overall_confidence,
                entry[0].casefold(),
            ),
        )
        return [
            RankedAccount(
                rank=position,
                company=name,
                verdict=result.verdict,
                weighted_score=result.weighted_score,
                overall_confidence=result.overall_confidence,
                lower_bound=round(self.lower_bound(result), 4),
                near_term_score=self._near_term(result),
            )
            for position, (name, result) in enumerate(ordered, start=1)
        ]
