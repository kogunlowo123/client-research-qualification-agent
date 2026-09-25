"""Test doubles for ports. Used only by the test suite."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from client_research_agent.models import Chunk, ChunkStrategy, DocumentType
from client_research_agent.services.ports import ChatMessage, FetchResult, LLMResponse, LLMUsage

Responder = Callable[[Sequence[ChatMessage]], str | Mapping[str, Any]]


@dataclass
class ScriptedLLM:
    """Deterministic LLM: routes on a keyword in the last message, or pops a queue."""

    routes: dict[str, Responder | str | Mapping[str, Any]] = field(default_factory=dict)
    queue: list[str | Mapping[str, Any]] = field(default_factory=list)
    default: str | Mapping[str, Any] = "{}"
    name: str = "scripted-llm"
    calls: list[list[ChatMessage]] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.name

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        haystack = "\n".join(m.content for m in messages)
        reply: Any = None
        if self.queue:
            reply = self.queue.pop(0)
        else:
            for keyword, responder in self.routes.items():
                if keyword in haystack:
                    reply = responder(messages) if callable(responder) else responder
                    break
        if reply is None:
            reply = self.default
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(
            content=content, model=self.name, usage=LLMUsage(len(haystack) // 4, len(content) // 4)
        )


@dataclass
class StaticFetcher:
    pages: dict[str, tuple[int, str, str]] = field(default_factory=dict)
    requested: list[str] = field(default_factory=list)

    def add(self, url: str, body: str, *, status: int = 200, content_type: str = "text/html") -> None:
        self.pages[url] = (status, content_type, body)

    def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
        self.requested.append(url)
        status, content_type, body = self.pages.get(url, (404, "text/plain", "not found"))
        return FetchResult(url=url, final_url=url, status_code=status, content_type=content_type, text=body)


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    company: str = "Acme Corp",
    doc_id: str = "doc-1",
    document_type: DocumentType = DocumentType.PRESS_RELEASE,
    publication_date: date | None = date(2026, 5, 1),
    parent_id: str | None = None,
    strategy: ChunkStrategy = ChunkStrategy.CHILD,
    index: int = 0,
    url: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        company=company,
        url=url or f"https://acme.example.com/news/{doc_id}",
        title=f"{company} announcement {doc_id}",
        document_type=document_type,
        source_domain="acme.example.com",
        chunk_index=index,
        strategy=strategy,
        parent_id=parent_id,
        publication_date=publication_date,
        confidence=0.8,
        token_count=len(text.split()),
    )
