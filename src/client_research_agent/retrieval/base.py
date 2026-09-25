"""Protocols and helpers shared by every retriever in the package."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.utils.errors import CircuitOpenError, OutputValidationError, TransientError

#: Failures of an LLM / model-serving dependency that trigger a deterministic fallback.
LLM_FAILURES: tuple[type[Exception], ...] = (OutputValidationError, TransientError, CircuitOpenError)

#: Failures of a remote retrieval dependency (embedding endpoint, vector index, store).
DEPENDENCY_FAILURES: tuple[type[Exception], ...] = (TransientError, CircuitOpenError, OutputValidationError)


@runtime_checkable
class ChunkRetriever(Protocol):
    """A component that returns ranked chunks for a query."""

    def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]: ...


def reindex(results: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Return copies whose ``rank`` matches their list position."""
    return [r if r.rank == i else r.model_copy(update={"rank": i}) for i, r in enumerate(results)]


def dedupe_by_chunk_id(results: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the first (best-ranked) occurrence of each chunk."""
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for result in results:
        if result.chunk.chunk_id in seen:
            continue
        seen.add(result.chunk.chunk_id)
        unique.append(result)
    return unique


def company_matches(chunk: Chunk, company: str | None) -> bool:
    return company is None or chunk.company.casefold() == company.casefold()
