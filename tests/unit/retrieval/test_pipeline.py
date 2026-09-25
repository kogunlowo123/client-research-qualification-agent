from __future__ import annotations

import pytest

from client_research_agent.config.settings import RetrievalSettings
from client_research_agent.models import Chunk, ChunkStrategy
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.retrieval.crag import RetrievalVerdict
from client_research_agent.retrieval.embeddings import HashingEmbeddingClient
from client_research_agent.retrieval.pipeline import RetrievalOutcome, RetrievalPipeline, Retriever
from client_research_agent.retrieval.reranking import LLMReranker
from client_research_agent.utils.errors import UpstreamServiceError
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.retrieval._corpus import corpus_chunks
from tests.unit.retrieval._stores import MemoryDocumentStore, MemoryVectorIndex
from tests.unit.retrieval.conftest import FailingLLM

EMBEDDER = HashingEmbeddingClient(256)


def _setup(corpus: list[Chunk]) -> tuple[MemoryVectorIndex, MemoryDocumentStore]:
    index = MemoryVectorIndex()
    index.upsert(corpus, EMBEDDER.embed([c.embedding_text for c in corpus]))
    store = MemoryDocumentStore()
    store.save_chunks(corpus)
    return index, store


def _with_parents() -> list[Chunk]:
    corpus = corpus_chunks()
    parent = make_chunk(
        "acme-parent",
        "Acme investor day. Acme completed the migration of 70 percent of workloads to Azure. More context.",
        company="Acme Corp",
        strategy=ChunkStrategy.PARENT,
    )
    child = corpus[0].model_copy(update={"parent_id": "acme-parent"})
    return [parent, child, *corpus[1:]]


class TestRetrievalPipeline:
    def test_end_to_end_without_llm(self) -> None:
        corpus = _with_parents()
        index, store = _setup(corpus)
        pipeline = RetrievalPipeline.from_components(
            embedder=EMBEDDER,
            vector_index=index,
            document_store=store,
            settings=RetrievalSettings(top_k=3),
            corpus=corpus,
        )
        assert isinstance(pipeline, Retriever)
        assert pipeline.corpus_size == len(corpus) - 1  # parent excluded from the lexical corpus
        outcome = pipeline.retrieve("Acme cloud migration and data center exit", company="Acme Corp")
        assert isinstance(outcome, RetrievalOutcome)
        assert 0 < len(outcome.chunks) <= 3
        assert set(outcome.chunk_ids) & {"acme-cloud-1", "acme-cloud-2"}
        assert all(r.chunk.company == "Acme Corp" for r in outcome.chunks)
        assert [r.rank for r in outcome.chunks] == list(range(len(outcome.chunks)))
        assert outcome.verdict is RetrievalVerdict.CORRECT
        assert outcome.corrective_actions == ["accept"]
        assert outcome.rewritten_query.startswith("Acme Corp")
        stages = [line.split(":", 1)[0] for line in outcome.trace]
        for stage in (
            "rewrite",
            "multi_query",
            "graph",
            "crag",
            "rerank",
            "mmr",
            "parent_expansion",
            "compression",
        ):
            assert stage in stages
        cloud_1 = next(r for r in outcome.chunks if r.chunk.chunk_id == "acme-cloud-1")
        assert cloud_1.chunk.metadata["parent_text"].startswith("Acme investor day")
        assert outcome.queries[0] == "Acme cloud migration and data center exit"
        assert get_metrics().counter("retrieval.requests") == 1

    def test_llm_components_and_llm_reranker(self) -> None:
        corpus = corpus_chunks()
        index, store = _setup(corpus)
        llm = ScriptedLLM(
            routes={
                "search-query-rewrite": {"query": "Globex AWS cloud migration", "keywords": ["aws"]},
                "multi-query-generation": {"queries": ["Globex Amazon Web Services agreement"]},
                "grade-relevance": {"relevant": True, "score": 0.8, "reason": "ok"},
                "listwise-rerank": {"scores": [{"id": "0", "score": 9}]},
            }
        )
        pipeline = RetrievalPipeline.from_components(
            embedder=EMBEDDER,
            vector_index=index,
            document_store=store,
            llm=llm,
            settings=RetrievalSettings(top_k=2, multi_query_count=1),
            corpus=corpus,
            use_llm_reranker=True,
            enable_graph=False,
        )
        assert isinstance(pipeline._reranker, LLMReranker)
        outcome = pipeline.retrieve("Globex move to AWS", company="Globex Industries", top_k=2)
        assert outcome.rewritten_query == "Globex AWS cloud migration"
        assert outcome.trace[0].startswith("rewrite:llm")
        assert "Globex Amazon Web Services agreement" in outcome.queries
        assert outcome.mean_relevance == pytest.approx(0.8)
        assert outcome.chunks[0].chunk.chunk_id.startswith("globex-cloud")
        assert not any(line.startswith("graph:") for line in outcome.trace)

    def test_degrades_when_llm_down(self, failing_llm: FailingLLM) -> None:
        corpus = corpus_chunks()
        index, _ = _setup(corpus)
        pipeline = RetrievalPipeline.from_components(
            embedder=EMBEDDER,
            vector_index=index,
            llm=failing_llm,
            settings=RetrievalSettings(top_k=3, rerank_enabled=False),
            corpus=corpus,
            use_llm_reranker=True,
        )
        outcome = pipeline.retrieve("Initech cybersecurity controls", company="Initech Holdings")
        assert outcome.chunk_ids[0] == "initech-sec-1"
        assert outcome.trace[0].startswith("rewrite:fallback")
        assert not any(line.startswith("rerank:") for line in outcome.trace)
        assert not any(line.startswith("parent_expansion") for line in outcome.trace)

    def test_knowledge_refresh_reingests_into_corpus(self) -> None:
        index, store = _setup([])
        new_chunk = corpus_chunks()[0]
        calls: list[str] = []

        def refresh(query: str) -> int:
            calls.append(query)
            store.save_chunks([new_chunk])
            index.upsert([new_chunk], EMBEDDER.embed([new_chunk.embedding_text]))
            return 1

        pipeline = RetrievalPipeline.from_components(
            embedder=EMBEDDER,
            vector_index=index,
            document_store=store,
            settings=RetrievalSettings(crag_max_corrections=1),
            knowledge_refresh=refresh,
        )
        assert pipeline.corpus_size == 0
        outcome = pipeline.retrieve("Acme cloud migration to Azure", company="Acme Corp")
        assert calls == ["Acme cloud migration to Azure"]
        assert "refresh" in outcome.corrective_actions
        assert outcome.chunk_ids == ["acme-cloud-1"]
        assert pipeline.corpus_size == 1

    def test_refresh_store_failure_is_tolerated(self) -> None:
        index, store = _setup([])

        class BrokenStore(MemoryDocumentStore):
            def list_chunks(self, company: str) -> list[Chunk]:
                raise UpstreamServiceError("delta table unavailable", status_code=503)

        pipeline = RetrievalPipeline.from_components(
            embedder=EMBEDDER,
            vector_index=index,
            document_store=BrokenStore(),
            settings=RetrievalSettings(crag_max_corrections=0),
            knowledge_refresh=lambda query: 2,
        )
        outcome = pipeline.retrieve("anything at all", company="Acme Corp")
        assert outcome.chunks == []
        assert outcome.verdict is RetrievalVerdict.INCORRECT
        assert store.chunks == {}

    def test_corpus_management_and_validation(self) -> None:
        corpus = corpus_chunks()
        index, _ = _setup(corpus)
        pipeline = RetrievalPipeline.from_components(
            embedder=EMBEDDER, vector_index=index, settings=RetrievalSettings(), corpus=corpus[:2]
        )
        assert pipeline.corpus_size == 2
        pipeline.add_to_corpus(corpus[1:4])
        assert pipeline.corpus_size == 4
        pipeline.refresh_corpus(corpus[:1])
        assert pipeline.corpus_size == 1
        with pytest.raises(ValueError, match="empty"):
            pipeline.retrieve("   ", company="Acme Corp")

    def test_default_settings_are_loaded(self) -> None:
        pipeline = RetrievalPipeline.from_components(embedder=EMBEDDER, vector_index=MemoryVectorIndex())
        assert pipeline.corpus_size == 0
