"""Small-to-big retrieval: match on child chunks, give the LLM the parent's context.

The child ``chunk_id`` stays the citation anchor (the child text is what was
matched and is short enough to quote); the parent's text is attached as
``metadata["parent_text"]`` for generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.base import DEPENDENCY_FAILURES
from client_research_agent.services.ports import DocumentStore

_logger = get_logger(__name__)


class ParentExpander:
    def __init__(
        self,
        document_store: DocumentStore,
        *,
        max_parent_chars: int = 8000,
        dedupe_parents: bool = True,
    ) -> None:
        if max_parent_chars < 1:
            raise ValueError("max_parent_chars must be >= 1")
        self._store = document_store
        self._max_chars = max_parent_chars
        self._dedupe = dedupe_parents

    @traced("parent_child.expand", span_type=SpanType.RETRIEVER)
    def expand(self, results: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        parent_ids = list(dict.fromkeys(r.chunk.parent_id for r in results if r.chunk.parent_id))
        if not parent_ids:
            return list(results)
        try:
            parents = {p.chunk_id: p for p in self._store.get_chunks(parent_ids)}
        except DEPENDENCY_FAILURES as exc:
            _logger.warning("parent_expansion_degraded", error=str(exc)[:200])
            get_metrics().increment("retrieval.parent_expansion.fallback")
            return list(results)
        expanded: list[RetrievedChunk] = []
        first_child_for: dict[str, str] = {}
        attached = 0
        for result in results:
            parent = parents.get(result.chunk.parent_id or "")
            if parent is None:
                expanded.append(result)
                continue
            metadata: dict[str, Any] = {**result.chunk.metadata, "parent_id": parent.chunk_id}
            anchor = first_child_for.get(parent.chunk_id)
            if self._dedupe and anchor is not None:
                metadata["parent_text_ref"] = anchor
            else:
                metadata["parent_text"] = self._clip(parent, result.chunk)
                first_child_for[parent.chunk_id] = result.chunk.chunk_id
                attached += 1
            expanded.append(
                result.model_copy(update={"chunk": result.chunk.model_copy(update={"metadata": metadata})})
            )
        get_metrics().increment("retrieval.parent_expansion.attached", attached)
        return expanded

    def _clip(self, parent: Chunk, child: Chunk) -> str:
        """Parent text bounded to ``max_parent_chars``, centred on the child when clipping."""
        text = parent.text
        if len(text) <= self._max_chars:
            return text
        position = text.find(child.text[:80])
        centre = position + len(child.text) // 2 if position >= 0 else 0
        start = max(0, min(centre - self._max_chars // 2, len(text) - self._max_chars))
        return text[start : start + self._max_chars]
