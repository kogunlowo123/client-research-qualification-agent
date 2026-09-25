"""Retrieval quality gates: recall@5 and MRR over a labelled multi-company corpus."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from client_research_agent.config.settings import RetrievalSettings
from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HashingEmbeddingClient,
    HybridRetriever,
    MultiQueryRetriever,
    RetrievalPipeline,
)
from tests.unit.retrieval._corpus import QUERIES, LabelledQuery, corpus_chunks
from tests.unit.retrieval._stores import MemoryDocumentStore, MemoryVectorIndex

pytestmark = pytest.mark.rag_eval

RECALL_AT_5_GATE = 0.8
MRR_GATE = 0.6

Search = Callable[[LabelledQuery], Sequence[RetrievedChunk]]


def _evaluate(search: Search, k: int = 5) -> tuple[float, float]:
    recall_total = 0.0
    rr_total = 0.0
    for labelled in QUERIES:
        ranked = [r.chunk.chunk_id for r in search(labelled)][:k]
        recall_total += len(labelled.relevant & set(ranked)) / len(labelled.relevant)
        rr_total += next((1.0 / (i + 1) for i, cid in enumerate(ranked) if cid in labelled.relevant), 0.0)
    return recall_total / len(QUERIES), rr_total / len(QUERIES)


@pytest.fixture(scope="module")
def corpus() -> list[Chunk]:
    return corpus_chunks()


@pytest.fixture(scope="module")
def embedder() -> HashingEmbeddingClient:
    return HashingEmbeddingClient(dimension=512)


@pytest.fixture(scope="module")
def index(corpus: list[Chunk], embedder: HashingEmbeddingClient) -> MemoryVectorIndex:
    vector_index = MemoryVectorIndex()
    vector_index.upsert(corpus, embedder.embed([c.embedding_text for c in corpus]))
    return vector_index


@pytest.fixture(scope="module")
def hybrid(
    corpus: list[Chunk], embedder: HashingEmbeddingClient, index: MemoryVectorIndex
) -> HybridRetriever:
    return HybridRetriever.from_settings(
        BM25Retriever(corpus), DenseRetriever(embedder, index), RetrievalSettings()
    )


def test_corpus_is_realistically_sized(corpus: list[Chunk]) -> None:
    assert 20 <= len(corpus) <= 40
    assert len({c.company for c in corpus}) == 3


def test_hybrid_company_filtered_meets_quality_gates(hybrid: HybridRetriever) -> None:
    recall, mrr = _evaluate(lambda q: hybrid.retrieve(q.query, k=5, company=q.company))
    assert recall >= RECALL_AT_5_GATE, f"recall@5={recall:.3f}"
    assert mrr >= MRR_GATE, f"mrr={mrr:.3f}"


def test_hybrid_unfiltered_meets_quality_gates(hybrid: HybridRetriever) -> None:
    recall, mrr = _evaluate(lambda q: hybrid.retrieve(q.query, k=5))
    assert recall >= RECALL_AT_5_GATE, f"recall@5={recall:.3f}"
    assert mrr >= MRR_GATE, f"mrr={mrr:.3f}"


def test_hybrid_is_at_least_as_good_as_each_leg(
    corpus: list[Chunk], embedder: HashingEmbeddingClient, index: MemoryVectorIndex, hybrid: HybridRetriever
) -> None:
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(embedder, index)
    hybrid_recall, _ = _evaluate(lambda q: hybrid.retrieve(q.query, k=5, company=q.company))
    bm25_recall, _ = _evaluate(lambda q: bm25.retrieve(q.query, k=5, company=q.company))
    dense_recall, _ = _evaluate(lambda q: dense.retrieve(q.query, k=5, company=q.company))
    assert hybrid_recall >= min(bm25_recall, dense_recall)


def test_multi_query_hybrid_meets_quality_gates(hybrid: HybridRetriever) -> None:
    multi = MultiQueryRetriever(hybrid, None, query_count=3)
    recall, mrr = _evaluate(lambda q: multi.retrieve(q.query, k=5, company=q.company))
    assert recall >= RECALL_AT_5_GATE, f"recall@5={recall:.3f}"
    assert mrr >= MRR_GATE, f"mrr={mrr:.3f}"


def test_full_pipeline_meets_quality_gates(
    corpus: list[Chunk], embedder: HashingEmbeddingClient, index: MemoryVectorIndex
) -> None:
    store = MemoryDocumentStore()
    store.save_chunks(corpus)
    pipeline = RetrievalPipeline.from_components(
        embedder=embedder,
        vector_index=index,
        document_store=store,
        settings=RetrievalSettings(top_k=5),
        corpus=corpus,
    )
    recall, mrr = _evaluate(lambda q: pipeline.retrieve(q.query, company=q.company, top_k=5).chunks)
    assert recall >= RECALL_AT_5_GATE, f"recall@5={recall:.3f}"
    assert mrr >= MRR_GATE, f"mrr={mrr:.3f}"
