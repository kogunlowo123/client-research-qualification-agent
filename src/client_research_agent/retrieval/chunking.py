"""Document chunking strategies.

All chunkers work on character *spans* of the source text rather than on
copied strings, so every chunk records exactly where it came from
(``metadata["char_start"]``/``["char_end"]``). That keeps quotes verifiable
against the original document and lets the enricher find the nearest section
heading for contextual retrieval.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from client_research_agent.config.settings import ChunkingSettings
from client_research_agent.models import Chunk, ChunkStrategy, SourceDocument
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.base import DEPENDENCY_FAILURES
from client_research_agent.retrieval.embeddings import embed_in_batches
from client_research_agent.retrieval.lexical import sentence_spans
from client_research_agent.retrieval.tokenization import Tokenizer, get_tokenizer
from client_research_agent.services.ports import EmbeddingClient

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

Span = tuple[int, int]

_logger = get_logger(__name__)


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: SourceDocument) -> list[Chunk]: ...


def make_chunk_id(doc_id: str, strategy: ChunkStrategy, index: int) -> str:
    """Deterministic chunk id: re-chunking the same document yields the same ids."""
    return hashlib.sha256(f"{doc_id}|{strategy.value}|{index}".encode()).hexdigest()[:24]


def build_chunk(
    document: SourceDocument,
    span: Span,
    *,
    index: int,
    strategy: ChunkStrategy,
    tokenizer: Tokenizer,
    parent_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Chunk | None:
    """Materialise a chunk for ``document.text[span]``; ``None`` if the span is blank."""
    start, end = span
    raw = document.text[start:end]
    text = raw.strip()
    if not text:
        return None
    char_start = start + (len(raw) - len(raw.lstrip()))
    metadata: dict[str, Any] = {
        "char_start": char_start,
        "char_end": char_start + len(text),
        "language": document.language,
        "content_hash": document.content_hash,
        **(extra_metadata or {}),
    }
    return Chunk(
        chunk_id=make_chunk_id(document.doc_id, strategy, index),
        doc_id=document.doc_id,
        text=text,
        company=document.company,
        url=document.url,
        title=document.title,
        document_type=document.document_type,
        source_domain=document.source_domain,
        chunk_index=index,
        strategy=strategy,
        parent_id=parent_id,
        publication_date=document.publication_date,
        industry=document.industry,
        confidence=document.trust_score,
        token_count=tokenizer.count(text),
        metadata=metadata,
    )


class SpanSplitter:
    """Recursive separator splitting of a text span into token-bounded spans with overlap."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self._tok = tokenizer
        self._max = max_tokens
        self._overlap = overlap_tokens
        self._budget = max_tokens - overlap_tokens
        self._separators = tuple(s for s in separators if s)

    def split(self, text: str, start: int = 0, end: int | None = None) -> list[Span]:
        stop = len(text) if end is None else end
        if not text[start:stop].strip():
            return []
        pieces = self._atomic(text, start, stop, 0)
        merged = self._merge(text, pieces)
        if self._overlap == 0:
            return merged
        spans: list[Span] = []
        for i, (s, e) in enumerate(merged):
            spans.append((self._overlap_start(text, merged[i - 1], s) if i else s, e))
        return spans

    def _count(self, text: str, start: int, end: int) -> int:
        return self._tok.count(text[start:end])

    def _atomic(self, text: str, start: int, end: int, level: int) -> list[Span]:
        if self._count(text, start, end) <= self._budget:
            return [(start, end)]
        for depth in range(level, len(self._separators)):
            separator = self._separators[depth]
            if text.find(separator, start, end) == -1:
                continue
            spans: list[Span] = []
            for s, e in _split_on(text, start, end, separator):
                spans.extend(self._atomic(text, s, e, depth + 1))
            return spans
        return self._hard_split(text, start, end)

    def _hard_split(self, text: str, start: int, end: int) -> list[Span]:
        spans: list[Span] = []
        group_start: int | None = None
        group_end = start
        for s, e in self._tok.word_spans(text, start, end):
            if self._count(text, s, e) > self._budget:
                if group_start is not None:
                    spans.append((group_start, group_end))
                    group_start = None
                spans.extend(self._char_split(text, s, e))
                continue
            if group_start is None:
                group_start = s
            elif self._count(text, group_start, e) > self._budget:
                spans.append((group_start, group_end))
                group_start = s
            group_end = e
        if group_start is not None:
            spans.append((group_start, group_end))
        return spans

    def _char_split(self, text: str, start: int, end: int) -> list[Span]:
        spans: list[Span] = []
        cursor = start
        while cursor < end:
            width = min(end - cursor, max(1, self._budget * 4))
            while width > 1 and self._count(text, cursor, cursor + width) > self._budget:
                width = max(1, width * 3 // 4)
            spans.append((cursor, cursor + width))
            cursor += width
        return spans

    def _merge(self, text: str, pieces: Sequence[Span]) -> list[Span]:
        merged: list[Span] = []
        current: Span | None = None
        for piece in pieces:
            if not text[piece[0] : piece[1]].strip():
                if current is not None:
                    current = (current[0], piece[1])
                continue
            if current is None:
                current = piece
            elif self._count(text, current[0], piece[1]) <= self._budget:
                current = (current[0], piece[1])
            else:
                merged.append(current)
                current = piece
        if current is not None:
            merged.append(current)
        return merged

    def _overlap_start(self, text: str, previous: Span, default: int) -> int:
        words = self._tok.word_spans(text, previous[0], previous[1])
        start = default
        for s, _ in reversed(words):
            if self._count(text, s, previous[1]) > self._overlap:
                break
            start = s
        return start


def _split_on(text: str, start: int, end: int, separator: str) -> list[Span]:
    spans: list[Span] = []
    cursor = start
    while True:
        index = text.find(separator, cursor, end)
        if index == -1:
            break
        cut = index + len(separator)
        spans.append((cursor, cut))
        cursor = cut
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _materialise(
    document: SourceDocument,
    spans: Sequence[Span],
    *,
    strategy: ChunkStrategy,
    tokenizer: Tokenizer,
    first_index: int = 0,
    parent_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for span in spans:
        chunk = build_chunk(
            document,
            span,
            index=first_index + len(chunks),
            strategy=strategy,
            tokenizer=tokenizer,
            parent_id=parent_id,
            extra_metadata=extra_metadata,
        )
        if chunk is not None:
            chunks.append(chunk)
    return chunks


class RecursiveChunker:
    """Split on paragraph, line, sentence then word boundaries within a token budget."""

    def __init__(
        self,
        *,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        tokenizer: Tokenizer | None = None,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        self._tok = tokenizer or get_tokenizer()
        self._splitter = SpanSplitter(
            self._tok, max_tokens=max_tokens, overlap_tokens=overlap_tokens, separators=separators
        )

    @classmethod
    def from_settings(
        cls, settings: ChunkingSettings, tokenizer: Tokenizer | None = None
    ) -> RecursiveChunker:
        return cls(
            max_tokens=settings.child_chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            tokenizer=tokenizer,
        )

    @traced("chunking.recursive", span_type=SpanType.PARSER)
    def chunk(self, document: SourceDocument) -> list[Chunk]:
        spans = self._splitter.split(document.text)
        chunks = _materialise(document, spans, strategy=ChunkStrategy.RECURSIVE, tokenizer=self._tok)
        get_metrics().increment("chunking.chunks", len(chunks), strategy=ChunkStrategy.RECURSIVE.value)
        return chunks


class SemanticChunker:
    """Break between sentences where embedding distance spikes above a percentile threshold."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        *,
        max_tokens: int = 256,
        breakpoint_percentile: float = 90.0,
        tokenizer: Tokenizer | None = None,
        window: int = 1,
        batch_size: int = 64,
    ) -> None:
        if not 0.0 < breakpoint_percentile < 100.0:
            raise ValueError("breakpoint_percentile must be in (0, 100)")
        if window < 0:
            raise ValueError("window must be >= 0")
        self._embedder = embedder
        self._max = max_tokens
        self._percentile = breakpoint_percentile
        self._tok = tokenizer or get_tokenizer()
        self._window = window
        self._batch_size = batch_size
        self._splitter = SpanSplitter(self._tok, max_tokens=max_tokens, overlap_tokens=0)

    @classmethod
    def from_settings(
        cls, embedder: EmbeddingClient, settings: ChunkingSettings, tokenizer: Tokenizer | None = None
    ) -> SemanticChunker:
        return cls(
            embedder,
            max_tokens=settings.child_chunk_tokens,
            breakpoint_percentile=settings.semantic_breakpoint_percentile,
            tokenizer=tokenizer,
        )

    @traced("chunking.semantic", span_type=SpanType.PARSER)
    def chunk(self, document: SourceDocument) -> list[Chunk]:
        text = document.text
        sentences = sentence_spans(text)
        if not sentences:
            return []
        try:
            groups = self._group(text, sentences)
            fallback = False
        except DEPENDENCY_FAILURES as exc:
            _logger.warning("semantic_chunking_fallback", doc_id=document.doc_id, error=str(exc)[:200])
            get_metrics().increment("chunking.semantic_fallback")
            groups = self._splitter.split(text)
            fallback = True
        spans: list[Span] = []
        for start, end in groups:
            if self._tok.count(text[start:end]) > self._max:
                spans.extend(self._splitter.split(text, start, end))
            else:
                spans.append((start, end))
        chunks = _materialise(
            document,
            spans,
            strategy=ChunkStrategy.SEMANTIC,
            tokenizer=self._tok,
            extra_metadata={"semantic_fallback": fallback},
        )
        get_metrics().increment("chunking.chunks", len(chunks), strategy=ChunkStrategy.SEMANTIC.value)
        return chunks

    def breakpoints(self, text: str, sentences: Sequence[Span]) -> set[int]:
        """Indices ``i`` such that a chunk boundary falls before sentence ``i``."""
        if len(sentences) < 2:
            return set()
        windows = []
        for i in range(len(sentences)):
            lo = max(0, i - self._window)
            hi = min(len(sentences) - 1, i + self._window)
            windows.append(text[sentences[lo][0] : sentences[hi][1]])
        vectors = np.asarray(embed_in_batches(self._embedder, windows, self._batch_size), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0.0] = 1.0
        unit = vectors / norms[:, None]
        distances = 1.0 - np.sum(unit[:-1] * unit[1:], axis=1)
        threshold = float(np.percentile(distances, self._percentile))
        return {i + 1 for i, distance in enumerate(distances) if float(distance) > threshold}

    def _group(self, text: str, sentences: Sequence[Span]) -> list[Span]:
        breaks = self.breakpoints(text, sentences)
        groups: list[Span] = []
        current: Span | None = None
        for i, (start, end) in enumerate(sentences):
            if current is None:
                current = (start, end)
            elif i in breaks or self._tok.count(text[current[0] : end]) > self._max:
                groups.append(current)
                current = (start, end)
            else:
                current = (current[0], end)
        if current is not None:
            groups.append(current)
        return groups


class ParentChildChunker:
    """Large parent chunks for LLM context, small child chunks (with ``parent_id``) for retrieval."""

    def __init__(
        self,
        *,
        parent_tokens: int = 1024,
        child_tokens: int = 256,
        overlap_tokens: int = 32,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if parent_tokens <= child_tokens:
            raise ValueError("parent_tokens must exceed child_tokens")
        self._tok = tokenizer or get_tokenizer()
        self._parent_splitter = SpanSplitter(self._tok, max_tokens=parent_tokens, overlap_tokens=0)
        self._child_splitter = SpanSplitter(self._tok, max_tokens=child_tokens, overlap_tokens=overlap_tokens)

    @classmethod
    def from_settings(
        cls, settings: ChunkingSettings, tokenizer: Tokenizer | None = None
    ) -> ParentChildChunker:
        return cls(
            parent_tokens=settings.parent_chunk_tokens,
            child_tokens=settings.child_chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            tokenizer=tokenizer,
        )

    @traced("chunking.parent_child", span_type=SpanType.PARSER)
    def split(self, document: SourceDocument) -> tuple[list[Chunk], list[Chunk]]:
        """Return ``(parents, children)``; every child's ``parent_id`` names its parent."""
        text = document.text
        parents: list[Chunk] = []
        children: list[Chunk] = []
        for span in self._parent_splitter.split(text):
            parent_index = len(parents)
            parent_id = make_chunk_id(document.doc_id, ChunkStrategy.PARENT, parent_index)
            kids = _materialise(
                document,
                self._child_splitter.split(text, span[0], span[1]),
                strategy=ChunkStrategy.CHILD,
                tokenizer=self._tok,
                first_index=len(children),
                parent_id=parent_id,
                extra_metadata={"parent_index": parent_index},
            )
            parent = build_chunk(
                document,
                span,
                index=parent_index,
                strategy=ChunkStrategy.PARENT,
                tokenizer=self._tok,
                extra_metadata={"child_ids": [kid.chunk_id for kid in kids]},
            )
            if parent is None:
                continue
            parents.append(parent)
            children.extend(kids)
        metrics = get_metrics()
        metrics.increment("chunking.chunks", len(parents), strategy=ChunkStrategy.PARENT.value)
        metrics.increment("chunking.chunks", len(children), strategy=ChunkStrategy.CHILD.value)
        return parents, children

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        parents, children = self.split(document)
        return [*parents, *children]
