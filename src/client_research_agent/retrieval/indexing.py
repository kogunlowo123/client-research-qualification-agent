"""Indexing: documents -> parent/child chunks -> enrichment -> embeddings -> index.

Parents are persisted to the document store only (they are the LLM context
unit, too large to embed usefully). Children are persisted *and* embedded via
their contextual ``embedding_text`` and upserted into the vector index.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from client_research_agent.config.settings import ChunkingSettings
from client_research_agent.models import Chunk, SourceDocument
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.chunking import ParentChildChunker
from client_research_agent.retrieval.embeddings import embed_in_batches
from client_research_agent.retrieval.enrichment import MetadataEnricher
from client_research_agent.retrieval.tokenization import Tokenizer
from client_research_agent.services.ports import DocumentStore, EmbeddingClient, VectorIndex

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    chunks: list[Chunk] = field(default_factory=list)
    parents: list[Chunk] = field(default_factory=list)
    children_indexed: int = 0
    documents: int = 0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks) + len(self.parents)


class IndexingPipeline:
    def __init__(
        self,
        *,
        chunking: ChunkingSettings,
        enricher: MetadataEnricher,
        embedder: EmbeddingClient,
        vector_index: VectorIndex,
        document_store: DocumentStore,
        batch_size: int = 64,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._chunker = ParentChildChunker.from_settings(chunking, tokenizer)
        self._enricher = enricher
        self._embedder = embedder
        self._index = vector_index
        self._store = document_store
        self._batch_size = batch_size

    @traced("indexing.index", span_type=SpanType.CHAIN)
    def index(self, documents: Sequence[SourceDocument]) -> IndexingResult:
        started = time.perf_counter()
        metrics = get_metrics()
        if not documents:
            return IndexingResult()
        parents: list[Chunk] = []
        children: list[Chunk] = []
        for document in documents:
            doc_parents, doc_children = self._chunker.split(document)
            parents.extend(self._enricher.enrich(doc_parents, document))
            children.extend(self._enricher.enrich(doc_children, document))
        self._store.save_documents(documents)
        self._store.save_chunks([*parents, *children])
        indexed = 0
        if children:
            vectors = embed_in_batches(self._embedder, [c.embedding_text for c in children], self._batch_size)
            indexed = self._index.upsert(children, vectors)
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics.increment("indexing.documents", len(documents))
        metrics.increment("indexing.parents", len(parents))
        metrics.increment("indexing.children", indexed)
        metrics.observe("indexing.latency_ms", elapsed_ms)
        _logger.info(
            "indexing_complete",
            documents=len(documents),
            parents=len(parents),
            children=len(children),
            indexed=indexed,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return IndexingResult(
            chunks=children, parents=parents, children_indexed=indexed, documents=len(documents)
        )
