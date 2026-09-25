"""Dense retrieval: embed the query, search the vector index with metadata filters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from client_research_agent.models import RetrievedChunk
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.services.ports import EmbeddingClient, VectorIndex
from client_research_agent.utils.errors import OutputValidationError

RETRIEVER_NAME = "dense"


class DenseRetriever:
    def __init__(self, embedder: EmbeddingClient, vector_index: VectorIndex) -> None:
        self._embedder = embedder
        self._index = vector_index

    @property
    def embedder(self) -> EmbeddingClient:
        return self._embedder

    @traced("dense.retrieve", span_type=SpanType.RETRIEVER)
    def retrieve(
        self,
        query: str,
        *,
        k: int,
        company: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip() or k <= 0:
            return []
        vectors = self._embedder.embed([query])
        if len(vectors) != 1:
            raise OutputValidationError(f"expected one query embedding, got {len(vectors)}")
        merged: dict[str, Any] = dict(filters or {})
        if company is not None:
            merged["company"] = company
        hits = self._index.search(vectors[0], k=k, filters=merged or None)
        get_metrics().increment("retrieval.dense.queries")
        return [
            RetrievedChunk(chunk=hit.chunk, score=hit.score, retriever=RETRIEVER_NAME, rank=rank)
            for rank, hit in enumerate(hits[:k])
        ]
