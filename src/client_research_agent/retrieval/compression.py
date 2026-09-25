"""Extractive context compression.

Keeps the sentences most relevant to the query and drops the rest, *verbatim*
and in original order. It never paraphrases, so every quote the briefing layer
cites can still be matched character-for-character against the source chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.lexical import sentence_spans, tokenize
from client_research_agent.retrieval.tokenization import count_tokens

_NUMERIC_BONUS = 0.1
SENTENCE_SEPARATOR = " "


class ContextCompressor:
    def __init__(self, *, max_sentences: int = 6, parent_max_sentences: int | None = None) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        self._max = max_sentences
        self._parent_max = parent_max_sentences or max_sentences * 2

    def select_sentences(self, query: str, text: str, max_sentences: int) -> list[str] | None:
        """Top sentences in document order, or ``None`` when nothing would be dropped."""
        spans = sentence_spans(text)
        if len(spans) <= max_sentences:
            return None
        query_terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for index, (start, end) in enumerate(spans):
            sentence = text[start:end]
            terms = set(tokenize(sentence))
            overlap = len(terms & query_terms) / len(query_terms) if query_terms else 0.0
            bonus = _NUMERIC_BONUS if overlap > 0 and any(ch.isdigit() for ch in sentence) else 0.0
            scored.append((overlap + bonus, index))
        relevant = [item for item in scored if item[0] > 0.0]
        if relevant:
            chosen = sorted(relevant, key=lambda item: (-item[0], item[1]))[:max_sentences]
            keep = sorted(index for _, index in chosen)
        else:
            keep = list(range(max_sentences))
        return [text[spans[i][0] : spans[i][1]] for i in keep]

    def compress_chunk(self, query: str, chunk: Chunk) -> Chunk:
        updates: dict[str, Any] = {}
        metadata: dict[str, Any] = dict(chunk.metadata)
        kept = self.select_sentences(query, chunk.text, self._max)
        if kept is not None:
            updates["text"] = SENTENCE_SEPARATOR.join(kept)
            updates["token_count"] = count_tokens(updates["text"])
            metadata["original_text_chars"] = len(chunk.text)
            metadata["kept_sentences"] = len(kept)
        parent_changed = False
        parent_text = metadata.get("parent_text")
        if isinstance(parent_text, str):
            kept_parent = self.select_sentences(query, parent_text, self._parent_max)
            if kept_parent is not None:
                metadata["parent_text"] = SENTENCE_SEPARATOR.join(kept_parent)
                metadata["parent_text_compressed"] = True
                parent_changed = True
        if kept is None and not parent_changed:
            return chunk
        metadata["compressed"] = True
        updates["metadata"] = metadata
        return chunk.model_copy(update=updates)

    @traced("compression.compress", span_type=SpanType.PARSER)
    def compress(self, query: str, results: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        compressed: list[RetrievedChunk] = []
        applied = 0
        for result in results:
            chunk = self.compress_chunk(query, result.chunk)
            if chunk is result.chunk:
                compressed.append(result)
                continue
            applied += 1
            compressed.append(result.model_copy(update={"chunk": chunk}))
        get_metrics().increment("retrieval.compression.applied", applied)
        return compressed
