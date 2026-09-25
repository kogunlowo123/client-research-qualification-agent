from __future__ import annotations

from collections.abc import Sequence

import pytest

from client_research_agent.config.settings import RetrievalSettings
from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.retrieval.base import ChunkRetriever, company_matches, dedupe_by_chunk_id, reindex
from client_research_agent.retrieval.bm25 import BM25Retriever, bm25_tokenize
from client_research_agent.retrieval.dense import DenseRetriever
from client_research_agent.retrieval.embeddings import HashingEmbeddingClient
from client_research_agent.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from client_research_agent.retrieval.multi_query import FALLBACK_TEMPLATES, MultiQueryRetriever
from client_research_agent.retrieval.query_rewriting import QueryRewriter, company_variants
from client_research_agent.utils.errors import OutputValidationError, UpstreamServiceError
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.retrieval._corpus import corpus_chunks
from tests.unit.retrieval._stores import MemoryVectorIndex
from tests.unit.retrieval.conftest import FailingLLM


def _hit(chunk_id: str, score: float = 1.0, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(chunk_id, f"text {chunk_id}"), score=score, retriever="t", rank=rank
    )


class StaticRetriever:
    def __init__(self, results: Sequence[RetrievedChunk], *, error: Exception | None = None) -> None:
        self.results = list(results)
        self.error = error
        self.queries: list[tuple[str, int, str | None]] = []

    def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
        self.queries.append((query, k, company))
        if self.error is not None:
            raise self.error
        return self.results[:k]


@pytest.fixture(scope="module")
def corpus() -> list[Chunk]:
    return corpus_chunks()


@pytest.fixture(scope="module")
def dense(corpus: list[Chunk]) -> DenseRetriever:
    embedder = HashingEmbeddingClient(256)
    index = MemoryVectorIndex()
    index.upsert(corpus, embedder.embed([c.embedding_text for c in corpus]))
    return DenseRetriever(embedder, index)


class TestBase:
    def test_helpers(self) -> None:
        hits = [_hit("a", rank=3), _hit("b", rank=1), _hit("a")]
        assert [h.rank for h in reindex(hits)] == [0, 1, 2]
        assert [h.chunk.chunk_id for h in dedupe_by_chunk_id(hits)] == ["a", "b"]
        chunk = make_chunk("x", "t", company="Acme Corp")
        assert company_matches(chunk, None)
        assert company_matches(chunk, "acme corp")
        assert not company_matches(chunk, "Globex")


class TestBM25:
    def test_ranks_and_filters(self, corpus: list[Chunk]) -> None:
        bm25 = BM25Retriever(corpus)
        assert isinstance(bm25, ChunkRetriever)
        assert len(bm25) == len(corpus)
        hits = bm25.retrieve("Acme cloud migration data centers", k=3, company="acme corp")
        assert hits[0].chunk.chunk_id in {"acme-cloud-1", "acme-cloud-2"}
        assert all(h.chunk.company == "Acme Corp" for h in hits)
        assert all(h.retriever == "bm25" for h in hits)
        assert [h.rank for h in hits] == [0, 1, 2]
        assert hits[0].score >= hits[-1].score
        assert get_metrics().counter("retrieval.bm25.queries") == 1

    def test_empty_cases(self, corpus: list[Chunk]) -> None:
        assert BM25Retriever().retrieve("anything", k=5) == []
        bm25 = BM25Retriever(corpus)
        assert bm25.retrieve("the of and", k=5) == []
        assert bm25.retrieve("cloud", k=0) == []
        assert bm25.retrieve("zzzunmatched", k=5) == []
        assert bm25.retrieve("cloud", k=5, company="Unknown Co") == []
        assert BM25Retriever([make_chunk("e", "the and of")]).retrieve("cloud", k=1) == []

    def test_refresh_replaces_corpus(self, corpus: list[Chunk]) -> None:
        bm25 = BM25Retriever(corpus[:2])
        bm25.refresh([*corpus, corpus[0]])
        assert len(bm25) == len(corpus)
        assert bm25.chunks[0].chunk_id == corpus[0].chunk_id
        assert bm25_tokenize("Migrating clouds") == ["migrat", "cloud"]


class TestDense:
    def test_company_filter_is_passed_to_index(self, dense: DenseRetriever) -> None:
        hits = dense.retrieve(
            "Globex cloud migration AWS", k=3, company="Globex Industries", filters={"x": 1}
        )
        assert dense.embedder.dimension == 256
        index = dense._index
        assert isinstance(index, MemoryVectorIndex)
        assert index.searches[-1] == {"x": 1, "company": "Globex Industries"}
        assert hits == []  # no chunk has attribute x == 1

    def test_ranks_relevant_first(self, dense: DenseRetriever) -> None:
        hits = dense.retrieve("Globex cloud migration AWS", k=3, company="Globex Industries")
        assert hits[0].chunk.chunk_id in {"globex-cloud-1", "globex-cloud-2"}
        assert all(h.retriever == "dense" for h in hits)
        assert dense.retrieve("   ", k=3) == []
        assert dense.retrieve("cloud", k=0) == []

    def test_rejects_wrong_embedding_count(self) -> None:
        class NoVectors:
            model_name = "none"
            dimension = 4

            def embed(self, texts: Sequence[str]) -> list[list[float]]:
                return []

        with pytest.raises(OutputValidationError):
            DenseRetriever(NoVectors(), MemoryVectorIndex()).retrieve("q", k=1)


class TestRRF:
    def test_weighted_fusion_and_dedupe(self) -> None:
        first = [_hit("a"), _hit("b"), _hit("a")]
        second = [_hit("b"), _hit("c")]
        fused = reciprocal_rank_fusion([first, second], k=60)
        assert [r.chunk.chunk_id for r in fused] == ["b", "a", "c"]
        assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
        weighted = reciprocal_rank_fusion(
            [first, second], k=60, weights=[1.0, 0.0], top_n=2, retriever_name="w"
        )
        assert [r.chunk.chunk_id for r in weighted] == ["a", "b"]
        assert weighted[0].retriever == "w"
        assert reciprocal_rank_fusion([]) == []

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            reciprocal_rank_fusion([[]], k=0)
        with pytest.raises(ValueError, match="weights"):
            reciprocal_rank_fusion([[]], weights=[1.0, 2.0])


class TestHybrid:
    def test_combines_both_legs(self, corpus: list[Chunk], dense: DenseRetriever) -> None:
        hybrid = HybridRetriever.from_settings(BM25Retriever(corpus), dense, RetrievalSettings())
        hits = hybrid.retrieve("Initech mainframe Google Cloud", k=4, company="Initech Holdings")
        assert hits[0].chunk.chunk_id == "initech-cloud-1"
        assert all(h.retriever == "hybrid" for h in hits)
        assert len({h.chunk.chunk_id for h in hits}) == len(hits) <= 4

    def test_dense_outage_degrades_to_sparse(self) -> None:
        sparse = StaticRetriever([_hit("a"), _hit("b")])
        down = StaticRetriever([], error=UpstreamServiceError("vector search 503", status_code=503))
        hybrid = HybridRetriever(sparse, down, candidate_pool=5)
        assert [h.chunk.chunk_id for h in hybrid.retrieve("q", k=2)] == ["a", "b"]
        assert sparse.queries == [("q", 5, None)]
        assert get_metrics().counter("retrieval.hybrid.dense_fallback") == 1

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="dense_weight"):
            HybridRetriever(StaticRetriever([]), StaticRetriever([]), dense_weight=1.5)


class TestQueryRewriter:
    def test_company_variants(self) -> None:
        assert company_variants("Acme Corp.") == ["Acme Corp.", "Acme"]
        assert company_variants("Globex Industries") == ["Globex Industries"]
        assert company_variants("Initech Holdings, Inc.") == ["Initech Holdings, Inc.", "Initech"]
        assert company_variants("  ") == []

    def test_deterministic_expansion(self) -> None:
        rewriter = QueryRewriter()
        rewrite = rewriter.rewrite("What is the AI and cloud migration strategy?", company="Acme Corp")
        assert rewrite.source == "fallback"
        assert rewrite.rewritten.startswith("Acme Corp ai cloud migration strategy")
        for term in ("strategy", "artificial", "machine", "aws", "azure"):
            assert term in rewrite.rewritten
        assert rewrite.hypothetical is None
        assert rewrite.keywords == ("ai", "cloud", "migration", "strategy")
        assert rewriter.expand("the of", company=None) == "the of"

    def test_llm_rewrite_and_company_injection(self) -> None:
        llm = ScriptedLLM(
            routes={
                "search-query-rewrite": {
                    "query": "cloud migration Azure data center exit",
                    "keywords": ["cloud"],
                },
                "hyde-passage": {"passage": "Acme   moved workloads to Azure."},
            }
        )
        rewrite = QueryRewriter(llm, use_hyde=True).rewrite("cloud plans?", company="Acme Corp")
        assert rewrite.source == "llm"
        assert rewrite.rewritten == "Acme Corp cloud migration Azure data center exit"
        assert rewrite.keywords == ("cloud",)
        assert rewrite.hypothetical == "Acme moved workloads to Azure."

    def test_llm_failure_uses_fallbacks(self, failing_llm: FailingLLM) -> None:
        rewriter = QueryRewriter(failing_llm, use_hyde=True)
        rewrite = rewriter.rewrite("earnings outlook", company="Globex")
        assert rewrite.source == "fallback"
        assert rewrite.hypothetical is not None
        assert rewrite.hypothetical.startswith("Globex announced progress on earnings outlook.")
        assert "revenue" in rewrite.hypothetical
        assert get_metrics().counter("retrieval.rewrite.fallback") == 1
        assert get_metrics().counter("retrieval.hyde.fallback") == 1

    def test_blank_llm_output_falls_back(self) -> None:
        llm = ScriptedLLM(routes={"search-query-rewrite": {"query": '"  "', "keywords": []}})
        assert QueryRewriter(llm).rewrite("cloud", company=None).source == "fallback"
        assert (
            QueryRewriter().fallback_passage("the", company=None) == "The company announced progress on the."
        )

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            QueryRewriter().rewrite("  ")

    def test_correction_queries_are_distinct(self) -> None:
        queries = QueryRewriter().correction_queries("AI investments", company="Acme Corp")
        assert len(queries) == 3
        assert len({q.casefold() for q in queries}) == 3
        assert "AI investments".casefold() not in {q.casefold() for q in queries}


class TestMultiQuery:
    def test_fallback_templates_fan_out_and_fuse(self) -> None:
        base = StaticRetriever([_hit("a"), _hit("b")])
        multi = MultiQueryRetriever(base, None, query_count=2, max_workers=2)
        result = multi.retrieve_with_queries(
            "cloud", k=2, company="Acme", extra_queries=["Acme cloud aws", "cloud"]
        )
        assert result.source == "fallback"
        assert result.queries == [
            "cloud",
            "Acme cloud aws",
            *[" ".join(t.format(company="Acme", query="cloud").split()) for t in FALLBACK_TEMPLATES[:2]],
        ]
        assert [h.chunk.chunk_id for h in result.chunks] == ["a", "b"]
        assert all(h.retriever == "multi_query" for h in result.chunks)
        assert len(multi.retrieve("cloud", k=1)) == 1

    def test_llm_queries(self) -> None:
        llm = ScriptedLLM(routes={"multi-query-generation": {"queries": ["q one", " ", "q two", "q three"]}})
        multi = MultiQueryRetriever(StaticRetriever([_hit("a")]), llm, query_count=2)
        queries, source = multi.generate_queries("orig", company=None)
        assert source == "llm"
        assert queries == ["q one", "q two"]

    def test_llm_failure_and_partial_subquery_failure(self, failing_llm: FailingLLM) -> None:
        class FlakyRetriever:
            def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
                if "announcement" in query:
                    raise UpstreamServiceError("timeout", status_code=504)
                return [_hit("a")]

        multi = MultiQueryRetriever(FlakyRetriever(), failing_llm, query_count=2)
        result = multi.retrieve_with_queries("cloud", k=3, company="Acme")
        assert result.source == "fallback"
        assert result.failed_queries == ["Acme cloud announcement"]
        assert [h.chunk.chunk_id for h in result.chunks] == ["a"]
        assert get_metrics().counter("retrieval.multi_query.failures") == 1

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="query_count"):
            MultiQueryRetriever(StaticRetriever([]), query_count=0)
