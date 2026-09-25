"""Minimal in-memory VectorIndex / DocumentStore for retrieval tests (test-only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from client_research_agent.models import Chunk, RetrievedChunk, SourceDocument


class MemoryVectorIndex:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[Chunk, np.ndarray]] = {}
        self.searches: list[Mapping[str, Any] | None] = []

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        for chunk, vector in zip(chunks, embeddings, strict=True):
            self.rows[chunk.chunk_id] = (chunk, np.asarray(vector, dtype=float))
        return len(chunks)

    def search(
        self, query_vector: Sequence[float], *, k: int, filters: Mapping[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.searches.append(filters)
        query = np.asarray(query_vector, dtype=float)
        scored: list[tuple[float, Chunk]] = []
        for chunk, vector in self.rows.values():
            if filters and not all(_matches(chunk, key, value) for key, value in filters.items()):
                continue
            denom = float(np.linalg.norm(query) * np.linalg.norm(vector)) or 1.0
            scored.append((float(np.dot(query, vector)) / denom, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievedChunk(chunk=chunk, score=score, retriever="vector", rank=rank)
            for rank, (score, chunk) in enumerate(scored[:k])
        ]

    def delete_company(self, company: str) -> int:
        doomed = [cid for cid, (c, _) in self.rows.items() if c.company.casefold() == company.casefold()]
        for cid in doomed:
            del self.rows[cid]
        return len(doomed)


def _matches(chunk: Chunk, key: str, value: Any) -> bool:
    actual = getattr(chunk, key, chunk.metadata.get(key))
    if isinstance(actual, str) and isinstance(value, str):
        return actual.casefold() == value.casefold()
    return bool(actual == value)


class MemoryDocumentStore:
    def __init__(self) -> None:
        self.documents: dict[str, SourceDocument] = {}
        self.chunks: dict[str, Chunk] = {}
        self.fail_get = False

    def save_documents(self, documents: Sequence[SourceDocument]) -> int:
        for document in documents:
            self.documents[document.doc_id] = document
        return len(documents)

    def save_chunks(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        return len(chunks)

    def list_chunks(self, company: str) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.company.casefold() == company.casefold()]

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if self.fail_get:
            from client_research_agent.utils.errors import UpstreamServiceError

            raise UpstreamServiceError("store unavailable", status_code=503)
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def known_hashes(self, company: str) -> set[str]:
        return {d.content_hash for d in self.documents.values() if d.company.casefold() == company.casefold()}
