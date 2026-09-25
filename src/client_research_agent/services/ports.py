"""Hexagonal ports.

Every external system (Databricks Model Serving, Vector Search, Unity Catalog
tables, the public web) is reached through one of these protocols. Production
wiring binds Databricks adapters; local development and CI bind the in-process
adapters in ``client_research_agent.services.local``. Business logic never
imports a vendor SDK directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from client_research_agent.models import Chunk, ClientBrief, RetrievedChunk, SourceDocument

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: str = "stop"


@runtime_checkable
class LLMClient(Protocol):
    """Chat-completions contract (Databricks Foundation Model API is OpenAI-compatible)."""

    @property
    def model_name(self) -> str: ...

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse: ...


@runtime_checkable
class EmbeddingClient(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Dense index over chunk embeddings (Databricks Vector Search in production)."""

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int: ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    def delete_company(self, company: str) -> int: ...


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@runtime_checkable
class HttpFetcher(Protocol):
    def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult: ...


@runtime_checkable
class DocumentStore(Protocol):
    """System of record for documents and chunks (Unity Catalog Delta tables in production)."""

    def save_documents(self, documents: Sequence[SourceDocument]) -> int: ...

    def save_chunks(self, chunks: Sequence[Chunk]) -> int: ...

    def list_chunks(self, company: str) -> list[Chunk]: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...

    def known_hashes(self, company: str) -> set[str]: ...


@runtime_checkable
class BriefRepository(Protocol):
    def save(self, brief: ClientBrief) -> str: ...

    def get(self, run_id: str) -> ClientBrief | None: ...


@runtime_checkable
class AuditSink(Protocol):
    def record(self, event_type: str, payload: Mapping[str, Any]) -> None: ...
