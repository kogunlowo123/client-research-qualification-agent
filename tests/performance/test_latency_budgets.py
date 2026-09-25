"""Latency budgets on local adapters (hashing embedder, in-memory index)."""

from __future__ import annotations

import random
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from client_research_agent.config.settings import AppSettings
from client_research_agent.models import Chunk, ChunkStrategy, DocumentType
from client_research_agent.orchestration import ClientResearchOrchestrator
from client_research_agent.retrieval.embeddings import HashingEmbeddingClient
from client_research_agent.retrieval.pipeline import RetrievalPipeline
from client_research_agent.services.local import InMemoryDocumentStore, InMemoryVectorIndex
from tests.support.world import ANALYST, local_runtime, northwind_request

pytestmark = pytest.mark.performance

#: Budgets are for uninstrumented code; line/branch coverage tracing slows pure-Python hot loops ~2x.
INSTRUMENTATION_FACTOR = 2.5


def budget(milliseconds: float) -> float:
    coverage = sys.modules.get("coverage")
    measured = coverage is not None and coverage.Coverage.current() is not None
    return milliseconds * INSTRUMENTATION_FACTOR if measured else milliseconds


COMPANY = "Contoso Manufacturing"
TOPICS = (
    "cloud migration of the ERP platform to Azure and decommissioning of two data centers",
    "generative AI assistant for field technicians built on a large language model",
    "annual revenue of {n} billion dollars across {m} countries and four segments",
    "appointed a new chief data officer to lead the enterprise data platform",
    "supply chain disruption and competitive pressure in industrial automation",
    "machine learning models for predictive maintenance across {m} plants",
    "cost reduction program and restructuring of the aftermarket business",
    "cybersecurity investment after regulatory changes in critical infrastructure",
)
QUERIES = (
    "Contoso Manufacturing revenue employees countries",
    "Contoso Manufacturing cloud migration ERP",
    "Contoso Manufacturing generative AI machine learning",
    "Contoso Manufacturing chief data officer appointment",
    "Contoso Manufacturing industry competition regulation",
    "Contoso Manufacturing budget restructuring cost reduction",
)


def synthetic_corpus(size: int) -> list[Chunk]:
    rng = random.Random(7)  # noqa: S311 - deterministic test data
    chunks: list[Chunk] = []
    for index in range(size):
        topic = TOPICS[index % len(TOPICS)].format(n=rng.randint(2, 40), m=rng.randint(5, 60))
        text = (
            f"{COMPANY} reported that it {topic}. Management discussed the program in the quarterly filing "
            f"number {index}, noting milestones for fiscal {2020 + index % 7} and plans for the next year."
        )
        chunks.append(
            Chunk(
                chunk_id=f"c{index:05d}",
                doc_id=f"d{index // 10:04d}",
                text=text,
                company=COMPANY,
                url=f"https://contoso.example.com/news/{index // 10}",
                title=f"{COMPANY} update {index // 10}",
                document_type=DocumentType.PRESS_RELEASE if index % 3 else DocumentType.SEC_FILING,
                source_domain="contoso.example.com",
                chunk_index=index % 10,
                strategy=ChunkStrategy.CHILD,
                publication_date=date(2026, 1, 1) - timedelta(days=index % 700),
                confidence=0.85,
                token_count=len(text.split()),
            )
        )
    return chunks


def test_retrieval_over_2000_chunks_p95_under_250ms(settings: AppSettings) -> None:
    corpus = synthetic_corpus(2000)
    embedder = HashingEmbeddingClient(512)
    index = InMemoryVectorIndex(512)
    index.upsert(corpus, embedder.embed([c.embedding_text for c in corpus]))
    store = InMemoryDocumentStore()
    store.save_chunks(corpus)
    pipeline = RetrievalPipeline.from_components(
        embedder=embedder,
        vector_index=index,
        document_store=store,
        settings=settings.retrieval,
        corpus=corpus,
    )
    pipeline.retrieve(QUERIES[0], company=COMPANY)  # warm-up (graph, BM25 caches)
    timings: list[float] = []
    for _ in range(4):
        for query in QUERIES:
            started = time.perf_counter()
            outcome = pipeline.retrieve(query, company=COMPANY)
            timings.append((time.perf_counter() - started) * 1000)
            assert outcome.chunks
    p95 = statistics.quantiles(timings, n=20)[18]
    assert p95 < budget(250.0), f"retrieval p95 {p95:.1f} ms"


def test_full_offline_orchestrator_run_under_10_seconds(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    started = time.perf_counter()
    result = ClientResearchOrchestrator(runtime).run(northwind_request(), principal=ANALYST)
    elapsed = time.perf_counter() - started
    assert result.persisted
    assert elapsed * 1000 < budget(10_000.0), f"orchestrator run took {elapsed:.2f} s"
