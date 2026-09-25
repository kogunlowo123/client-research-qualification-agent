"""Evidence ranking for qualification: quality first, then diversity.

Base score of a retrieved chunk::

    score = relevance * trust(document_type) * recency(publication_date)

where relevance is the retriever score normalised to ``[0, 1]`` across the
candidate batch, trust comes from ``SOURCE_TRUST`` and recency is an
exponential half-life decay.

Selection uses Maximal Marginal Relevance so the LLM sees corroborating
evidence from *different* documents rather than five near-identical chunks of
one press release::

    mmr = lambda * score_norm - (1 - lambda) * max_similarity_to_selected

Similarity blends content-word Jaccard overlap with same-document,
same-domain and same-type indicators. A per-domain cap spreads selections
across sources; it is a *soft* cap: when every remaining candidate is from a
capped domain (common for companies whose own site dominates the corpus) the
cap is relaxed rather than returning fewer than ``k`` items.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from client_research_agent.models import RetrievedChunk
from client_research_agent.qualification.evidence import normalize_relevance
from client_research_agent.qualification.signals import recency_weight, source_trust
from client_research_agent.qualification.text import content_words, jaccard


@dataclass(frozen=True, slots=True)
class RankedChunk:
    retrieved: RetrievedChunk
    relevance: float
    trust: float
    recency: float
    score: float
    mmr_score: float

    @property
    def chunk_id(self) -> str:
        return self.retrieved.chunk.chunk_id


class EvidenceRanker:
    def __init__(
        self,
        *,
        half_life_days: int = 365,
        mmr_lambda: float = 0.7,
        max_per_domain: int = 3,
        today: date | None = None,
    ) -> None:
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be within [0, 1]")
        if max_per_domain < 1:
            raise ValueError("max_per_domain must be >= 1")
        self._half_life_days = half_life_days
        self._lambda = mmr_lambda
        self._max_per_domain = max_per_domain
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    def base_scores(self, candidates: Sequence[RetrievedChunk]) -> list[RankedChunk]:
        """Deduplicate by chunk id (keeping the best retriever score) and compute base scores."""
        best: dict[str, RetrievedChunk] = {}
        for candidate in candidates:
            current = best.get(candidate.chunk.chunk_id)
            if current is None or candidate.score > current.score:
                best[candidate.chunk.chunk_id] = candidate
        unique = list(best.values())
        relevances = normalize_relevance([item.score for item in unique])
        ranked: list[RankedChunk] = []
        for item, relevance in zip(unique, relevances, strict=True):
            trust = source_trust(item.chunk.document_type)
            recency = recency_weight(
                item.chunk.publication_date, today=self.today, half_life_days=self._half_life_days
            )
            score = relevance * trust * recency
            ranked.append(RankedChunk(item, relevance, trust, recency, score, score))
        ranked.sort(key=lambda r: (-r.score, r.retrieved.rank, r.chunk_id))
        return ranked

    @staticmethod
    def similarity(left: RetrievedChunk, right: RetrievedChunk) -> float:
        a, b = left.chunk, right.chunk
        lexical = jaccard(content_words(a.text), content_words(b.text))
        return min(
            1.0,
            0.6 * lexical
            + (0.25 if a.doc_id == b.doc_id else 0.0)
            + (0.1 if a.source_domain == b.source_domain else 0.0)
            + (0.05 if a.document_type == b.document_type else 0.0),
        )

    def _mmr_value(self, item: RankedChunk, selected: Sequence[RankedChunk], top: float) -> float:
        redundancy = max((self.similarity(item.retrieved, s.retrieved) for s in selected), default=0.0)
        return self._lambda * (item.score / top) - (1 - self._lambda) * redundancy

    def rank(self, candidates: Sequence[RetrievedChunk], k: int) -> list[RankedChunk]:
        """Select up to ``k`` chunks by MMR with a soft per-domain cap, in selection order."""
        if k <= 0 or not candidates:
            return []
        pool = self.base_scores(candidates)
        top = pool[0].score or 1.0
        selected: list[RankedChunk] = []
        domains: Counter[str] = Counter()
        while pool and len(selected) < k:
            eligible = [r for r in pool if domains[r.retrieved.chunk.source_domain] < self._max_per_domain]
            if not eligible:
                eligible = pool
            scored = [(self._mmr_value(item, selected, top), item) for item in eligible]
            best_value, best_item = max(scored, key=lambda pair: pair[0])
            pool.remove(best_item)
            domains[best_item.retrieved.chunk.source_domain] += 1
            selected.append(
                RankedChunk(
                    best_item.retrieved,
                    best_item.relevance,
                    best_item.trust,
                    best_item.recency,
                    best_item.score,
                    round(best_value, 6),
                )
            )
        return selected
