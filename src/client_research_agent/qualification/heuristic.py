"""Deterministic, explainable criterion scorer.

The heuristic qualifier reads the same evidence the LLM sees and scores it
from lexicon hits weighted by source trust and recency, plus explicit
revenue/headcount magnitudes for the scale criterion. It never calls a model,
so it is the fallback whenever the LLM path fails and a cross-check on the
LLM's score when it succeeds.

Scoring for one criterion:

* each evidence item contributes ``w = trust * recency * (0.5 + 0.5 * relevance)``;
* positive mass ``P = sum(w * min(positive_hits, 3))``, negative mass ``N`` likewise;
* lexicon score ``round(5 * (1 - exp(-max(0, P - N) / 2.5)))``, so a single weak
  hit lands around 1 and several strong, recent, trusted hits approach 5;
* for company scale the score is the max of the lexicon score and the
  revenue/employee band, which are concrete and therefore preferred;
* confidence grows with the evidence mass and the number of distinct sources,
  capped at 0.8 because keyword matching can never be fully certain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from client_research_agent.config.settings import ScoringSettings
from client_research_agent.models import Criterion, CriterionScore
from client_research_agent.qualification.criteria import CriterionDefinition, get_definition
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.qualification.signals import (
    employee_band,
    extract_scale_signals,
    recency_weight,
    revenue_band,
    source_trust,
)
from client_research_agent.qualification.text import phrase_pattern

MAX_HEURISTIC_CONFIDENCE = 0.8
NO_EVIDENCE_CONFIDENCE = 0.05
_MAX_HITS_PER_ITEM = 3
_MAX_CITED = 6


@dataclass(frozen=True, slots=True)
class SignalHit:
    evidence_id: str
    weight: float
    positive: tuple[str, ...]
    negative: tuple[str, ...]

    @property
    def contribution(self) -> float:
        return self.weight * (
            min(len(self.positive), _MAX_HITS_PER_ITEM) - min(len(self.negative), _MAX_HITS_PER_ITEM)
        )


def _match_lexicon(text: str, lexicon: Sequence[str]) -> tuple[str, ...]:
    return tuple(phrase for phrase in lexicon if phrase_pattern(phrase).search(text))


class HeuristicQualifier:
    """Signal-based scorer producing a ``CriterionScore`` with a full reasoning trace."""

    def __init__(
        self,
        settings: ScoringSettings,
        *,
        today: date | None = None,
        half_life_days: int = 365,
    ) -> None:
        self._settings = settings
        self._today = today
        self._half_life_days = half_life_days

    @property
    def today(self) -> date:
        return self._today or date.today()

    def signal_hits(
        self, definition: CriterionDefinition, registry: EvidenceRegistry, evidence_ids: Sequence[str]
    ) -> list[SignalHit]:
        hits: list[SignalHit] = []
        for evidence_id in registry.partition_ids(evidence_ids)[0]:
            evidence = registry.require(evidence_id)
            text = registry.source_text(evidence_id) or evidence.quote
            weight = (
                source_trust(evidence.document_type)
                * recency_weight(
                    evidence.publication_date, today=self.today, half_life_days=self._half_life_days
                )
                * (0.5 + 0.5 * evidence.relevance)
            )
            hits.append(
                SignalHit(
                    evidence_id=evidence_id,
                    weight=weight,
                    positive=_match_lexicon(text, definition.positive_signals),
                    negative=_match_lexicon(text, definition.negative_signals),
                )
            )
        return hits

    def score(
        self, criterion: Criterion, registry: EvidenceRegistry, evidence_ids: Sequence[str]
    ) -> CriterionScore:
        definition = get_definition(criterion)
        weight = self._settings.weights[criterion]
        hits = self.signal_hits(definition, registry, evidence_ids)
        trace: list[str] = [f"heuristic: {len(hits)} evidence item(s) reviewed for {definition.title}"]
        if not hits:
            return CriterionScore(
                criterion=criterion,
                score=0,
                weight=weight,
                confidence=NO_EVIDENCE_CONFIDENCE,
                rationale=f"No public evidence was retrieved for {definition.title}.",
                evidence_ids=(),
                reasoning_trace=(*trace, "no evidence: score 0 per rubric level 0"),
            )

        positive_mass = sum(h.weight * min(len(h.positive), _MAX_HITS_PER_ITEM) for h in hits)
        negative_mass = sum(h.weight * min(len(h.negative), _MAX_HITS_PER_ITEM) for h in hits)
        net = max(0.0, positive_mass - negative_mass)
        lexicon_score = round(5 * (1 - math.exp(-net / 2.5)))
        for hit in hits:
            if hit.positive or hit.negative:
                trace.append(
                    f"{hit.evidence_id}: weight {hit.weight:.2f}; +{list(hit.positive)[:5]} "
                    f"-{list(hit.negative)[:5]}"
                )
        trace.append(
            f"positive mass {positive_mass:.2f}, negative mass {negative_mass:.2f}, lexicon score "
            f"{lexicon_score}"
        )

        final_score = lexicon_score
        numeric_ids: list[str] = []
        if criterion is Criterion.COMPANY_SCALE:
            final_score, numeric_ids = self._apply_scale_signals(registry, hits, lexicon_score, trace)

        contributing = sorted(
            (h for h in hits if h.positive or h.negative or h.evidence_id in numeric_ids),
            key=lambda h: (
                -(abs(h.contribution) + (1.0 if h.evidence_id in numeric_ids else 0.0)),
                h.evidence_id,
            ),
        )
        cited = tuple(h.evidence_id for h in contributing[:_MAX_CITED])
        distinct_sources = len({registry.require(h.evidence_id).url for h in hits})
        mass = positive_mass + negative_mass + (1.5 if numeric_ids else 0.0)
        confidence = 0.15 + 0.5 * (1 - math.exp(-mass / 2.0)) + (0.1 if distinct_sources > 1 else 0.0)
        if not cited:
            confidence = min(confidence, 0.25)
        confidence = round(min(MAX_HEURISTIC_CONFIDENCE, confidence), 3)
        trace.append(f"distinct sources {distinct_sources}; confidence {confidence:.2f}")

        rationale = self._rationale(definition, final_score, hits, numeric_ids)
        return CriterionScore(
            criterion=criterion,
            score=max(0, min(5, final_score)),
            weight=weight,
            confidence=confidence,
            rationale=rationale,
            evidence_ids=cited,
            reasoning_trace=tuple(trace),
        )

    def _apply_scale_signals(
        self,
        registry: EvidenceRegistry,
        hits: Sequence[SignalHit],
        lexicon_score: int,
        trace: list[str],
    ) -> tuple[int, list[str]]:
        best_band = 0
        numeric_ids: list[str] = []
        for hit in hits:
            signals = extract_scale_signals(registry.source_text(hit.evidence_id))
            bands: list[int] = []
            if signals.revenue_usd is not None:
                bands.append(revenue_band(signals.revenue_usd))
                trace.append(f"{hit.evidence_id}: revenue ~${signals.revenue_usd:,.0f} -> band {bands[-1]}")
            if signals.employees is not None:
                bands.append(employee_band(signals.employees))
                trace.append(f"{hit.evidence_id}: ~{signals.employees:,} employees -> band {bands[-1]}")
            if bands:
                numeric_ids.append(hit.evidence_id)
                best_band = max(best_band, *bands)
        if not numeric_ids:
            # Without any figure, keyword evidence alone cannot justify the top of the scale.
            capped = min(lexicon_score, 3)
            if capped != lexicon_score:
                trace.append("no revenue/headcount figures: lexicon score capped at 3")
            return capped, numeric_ids
        final = max(best_band, min(lexicon_score, best_band + 1))
        trace.append(f"scale band {best_band}; combined scale score {final}")
        return final, numeric_ids

    @staticmethod
    def _rationale(
        definition: CriterionDefinition, score: int, hits: Sequence[SignalHit], numeric_ids: Sequence[str]
    ) -> str:
        positives = sorted({phrase for h in hits for phrase in h.positive})
        negatives = sorted({phrase for h in hits for phrase in h.negative})
        parts = [f"Signal-based assessment of {definition.title}: level {score} - {definition.rubric[score]}"]
        if positives:
            parts.append(f"Supporting signals: {', '.join(positives[:8])}.")
        if negatives:
            parts.append(f"Countervailing signals: {', '.join(negatives[:6])}.")
        if numeric_ids:
            parts.append(f"Revenue/headcount figures found in {', '.join(numeric_ids[:4])}.")
        if not positives and not negatives and not numeric_ids:
            parts.append("Retrieved evidence contains no recognised signals for this criterion.")
        return " ".join(parts)
