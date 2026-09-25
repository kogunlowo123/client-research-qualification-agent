"""Structural contract between qualification and the retrieval layer.

Qualification depends only on these protocols, never on the concrete
retrieval pipeline, so either side can evolve independently and tests can
bind a simple in-memory retriever.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from client_research_agent.models import RetrievedChunk


@runtime_checkable
class RetrievalOutcomeLike(Protocol):
    @property
    def chunks(self) -> list[RetrievedChunk]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, *, company: str, top_k: int | None = None) -> RetrievalOutcomeLike: ...
