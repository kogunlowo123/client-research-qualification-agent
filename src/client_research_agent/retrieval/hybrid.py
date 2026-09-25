"""Hybrid retrieval with weighted Reciprocal Rank Fusion.

RRF fuses ranked lists using only ranks (``weight / (k + rank)``), so BM25 and
cosine scores on incomparable scales combine without calibration. If the dense
leg fails (embedding endpoint down, breaker open) retrieval degrades to BM25.
"""

from __future__ import annotations

from collections.abc import Sequence

from client_research_agent.config.settings import RetrievalSettings
from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.base import DEPENDENCY_FAILURES, ChunkRetriever

RETRIEVER_NAME = "hybrid"

_logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
    retriever_name: str = "rrf",
) -> list[RetrievedChunk]:
    """Weighted RRF over ranked lists; duplicates within a list count once (best rank)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if weights is not None and len(weights) != len(result_lists):
        raise ValueError("weights must align with result_lists")
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    first_seen: dict[str, int] = {}
    for list_index, results in enumerate(result_lists):
        weight = 1.0 if weights is None else float(weights[list_index])
        seen_here: set[str] = set()
        for position, result in enumerate(results):
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen_here:
                continue
            seen_here.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + position + 1)
            chunks.setdefault(chunk_id, result.chunk)
            first_seen.setdefault(chunk_id, len(first_seen))
    ordered = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return [
        RetrievedChunk(chunk=chunks[cid], score=scores[cid], retriever=retriever_name, rank=rank)
        for rank, cid in enumerate(ordered)
    ]


class HybridRetriever:
    """BM25 + dense retrieval fused with weighted RRF (``dense_weight`` vs ``1 - dense_weight``)."""

    def __init__(
        self,
        sparse: ChunkRetriever,
        dense: ChunkRetriever,
        *,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        candidate_pool: int = 40,
    ) -> None:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be in [0, 1]")
        self._sparse = sparse
        self._dense = dense
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._pool = candidate_pool

    @classmethod
    def from_settings(
        cls, sparse: ChunkRetriever, dense: ChunkRetriever, settings: RetrievalSettings
    ) -> HybridRetriever:
        return cls(
            sparse,
            dense,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            candidate_pool=settings.candidate_pool,
        )

    @traced("hybrid.retrieve", span_type=SpanType.RETRIEVER)
    def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
        pool = max(k, self._pool)
        sparse_hits = self._sparse.retrieve(query, k=pool, company=company)
        try:
            dense_hits = self._dense.retrieve(query, k=pool, company=company)
        except DEPENDENCY_FAILURES as exc:
            _logger.warning("dense_retrieval_degraded", error=str(exc)[:200])
            get_metrics().increment("retrieval.hybrid.dense_fallback")
            dense_hits = []
        return reciprocal_rank_fusion(
            [sparse_hits, dense_hits],
            k=self._rrf_k,
            weights=[1.0 - self._dense_weight, self._dense_weight],
            top_n=k,
            retriever_name=RETRIEVER_NAME,
        )
