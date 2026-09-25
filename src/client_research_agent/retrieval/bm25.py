"""Sparse lexical retrieval with Okapi BM25.

The corpus is an immutable snapshot swapped atomically on ``refresh`` so
concurrent queries (multi-query fan-out runs in a thread pool) never observe a
half-built index.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.lexical import tokenize

RETRIEVER_NAME = "bm25"


def bm25_tokenize(text: str) -> list[str]:
    """Analyzer used for both documents and queries (lowercase, stopwords, stemming)."""
    return tokenize(text)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    chunks: tuple[Chunk, ...]
    companies: tuple[str, ...]
    model: BM25Okapi | None


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk] = (), *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._lock = threading.Lock()
        self._snapshot = _Snapshot((), (), None)
        self.refresh(chunks)

    def __len__(self) -> int:
        return len(self._snapshot.chunks)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._snapshot.chunks

    def refresh(self, chunks: Sequence[Chunk]) -> None:
        """Rebuild the index over ``chunks`` (deduplicated by ``chunk_id``, last wins)."""
        unique = tuple({c.chunk_id: c for c in chunks}.values())
        corpus = [bm25_tokenize(c.embedding_text) for c in unique]
        model = BM25Okapi(corpus, k1=self._k1, b=self._b) if any(corpus) else None
        snapshot = _Snapshot(unique, tuple(c.company.casefold() for c in unique), model)
        with self._lock:
            self._snapshot = snapshot

    @traced("bm25.retrieve", span_type=SpanType.RETRIEVER)
    def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
        snapshot = self._snapshot
        terms = bm25_tokenize(query)
        if snapshot.model is None or not terms or k <= 0:
            return []
        scores = np.asarray(snapshot.model.get_scores(terms), dtype=np.float64)
        if company is not None:
            wanted = company.casefold()
            mask = np.fromiter(
                (c == wanted for c in snapshot.companies), dtype=bool, count=len(snapshot.companies)
            )
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")
        results: list[RetrievedChunk] = []
        for index in order:
            score = float(scores[index])
            if not np.isfinite(score) or score <= 0.0:
                break
            results.append(
                RetrievedChunk(
                    chunk=snapshot.chunks[int(index)],
                    score=score,
                    retriever=RETRIEVER_NAME,
                    rank=len(results),
                )
            )
            if len(results) >= k:
                break
        get_metrics().increment("retrieval.bm25.queries")
        return results
