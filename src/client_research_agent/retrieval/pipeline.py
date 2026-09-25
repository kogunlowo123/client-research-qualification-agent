"""The retrieval entry point used by research, qualification and briefing.

Flow for ``retrieve(query, company=...)``::

    rewrite (LLM | synonym expansion)
      -> multi-query hybrid search (BM25 + dense, weighted RRF)
      -> graph expansion over the entity co-occurrence graph (RRF-fused)
      -> CRAG: grade, correct (rewrite + re-retrieve), optional knowledge refresh
      -> rerank (lexical features | LLM listwise)
      -> MMR diversification
      -> parent expansion (small-to-big)
      -> extractive compression
      -> top_k

Every stage that depends on a model has a deterministic fallback, and each
stage appends a line to ``RetrievalOutcome.trace`` so the MLflow trace and the
audit record show exactly how the evidence was obtained.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from client_research_agent.config.settings import RetrievalSettings, get_settings
from client_research_agent.models import Chunk, ChunkStrategy, RetrievedChunk
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.bm25 import BM25Retriever
from client_research_agent.retrieval.compression import ContextCompressor
from client_research_agent.retrieval.crag import CorrectiveRetriever, RelevanceGrader, RetrievalVerdict
from client_research_agent.retrieval.dense import DenseRetriever
from client_research_agent.retrieval.graph import EntityGraph, GraphRetriever
from client_research_agent.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from client_research_agent.retrieval.multi_query import MultiQueryRetriever
from client_research_agent.retrieval.parent_child import ParentExpander
from client_research_agent.retrieval.query_rewriting import QueryRewriter
from client_research_agent.retrieval.reranking import (
    LexicalReranker,
    LLMReranker,
    Reranker,
    maximal_marginal_relevance,
)
from client_research_agent.services.ports import DocumentStore, EmbeddingClient, LLMClient, VectorIndex
from client_research_agent.utils.errors import AgentError

KnowledgeRefresh = Callable[[str], int]

#: Graph evidence is a recall booster, weighted below the text retrievers in fusion.
GRAPH_FUSION_WEIGHT = 0.5

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    query: str
    company: str
    chunks: list[RetrievedChunk]
    trace: list[str]
    corrective_actions: list[str]
    mean_relevance: float
    verdict: RetrievalVerdict = RetrievalVerdict.INCORRECT
    rewritten_query: str = ""
    queries: list[str] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [r.chunk.chunk_id for r in self.chunks]


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, *, company: str, top_k: int | None = None) -> RetrievalOutcome: ...


class _CandidateSearch:
    """Per-request first stage: multi-query hybrid search fused with graph expansion."""

    def __init__(
        self,
        multi_query: MultiQueryRetriever,
        graph: GraphRetriever | None,
        extra_queries: Sequence[str],
        trace: list[str],
    ) -> None:
        self._multi_query = multi_query
        self._graph = graph
        self._extra = list(extra_queries)
        self._trace = trace
        self.queries: list[str] = []

    def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
        extras, self._extra = self._extra, []  # rewrite extras apply to the first (original) query only
        result = self._multi_query.retrieve_with_queries(query, k=k, company=company, extra_queries=extras)
        self.queries.extend(q for q in result.queries if q not in self.queries)
        self._trace.append(
            f"multi_query:{result.source} queries={len(result.queries)} failed={len(result.failed_queries)} "
            f"hits={len(result.chunks)}"
        )
        if self._graph is None:
            return result.chunks
        graph_hits = self._graph.retrieve(query, k=max(1, k // 2), company=company)
        self._trace.append(f"graph:hits={len(graph_hits)}")
        if not graph_hits:
            return result.chunks
        return reciprocal_rank_fusion(
            [result.chunks, graph_hits],
            weights=[1.0, GRAPH_FUSION_WEIGHT],
            top_n=k,
            retriever_name="hybrid+graph",
        )


class RetrievalPipeline:
    def __init__(
        self,
        *,
        settings: RetrievalSettings,
        bm25: BM25Retriever,
        multi_query: MultiQueryRetriever,
        rewriter: QueryRewriter,
        grader: RelevanceGrader,
        reranker: Reranker,
        compressor: ContextCompressor,
        expander: ParentExpander | None = None,
        document_store: DocumentStore | None = None,
        enable_graph: bool = True,
        knowledge_refresh: KnowledgeRefresh | None = None,
        mmr_lambda: float = 0.7,
    ) -> None:
        self._settings = settings
        self._bm25 = bm25
        self._multi_query = multi_query
        self._rewriter = rewriter
        self._grader = grader
        self._reranker = reranker
        self._compressor = compressor
        self._expander = expander
        self._store = document_store
        self._enable_graph = enable_graph
        self._knowledge_refresh = knowledge_refresh
        self._mmr_lambda = mmr_lambda
        self._lock = threading.RLock()
        self._corpus: dict[str, Chunk] = {}
        self._graph: GraphRetriever | None = None
        self.refresh_corpus(bm25.chunks)

    @classmethod
    def from_components(
        cls,
        *,
        embedder: EmbeddingClient,
        vector_index: VectorIndex,
        document_store: DocumentStore | None = None,
        llm: LLMClient | None = None,
        settings: RetrievalSettings | None = None,
        corpus: Sequence[Chunk] = (),
        use_llm_reranker: bool = False,
        use_llm_grader: bool = True,
        use_hyde: bool = False,
        enable_graph: bool = True,
        knowledge_refresh: KnowledgeRefresh | None = None,
        mmr_lambda: float = 0.7,
        max_workers: int = 4,
    ) -> RetrievalPipeline:
        cfg = settings or get_settings().retrieval
        bm25 = BM25Retriever(corpus)
        hybrid = HybridRetriever.from_settings(bm25, DenseRetriever(embedder, vector_index), cfg)
        lexical = LexicalReranker(recency_half_life_days=cfg.recency_half_life_days)
        reranker: Reranker = (
            LLMReranker(llm, fallback=lexical) if (use_llm_reranker and llm is not None) else lexical
        )
        return cls(
            settings=cfg,
            bm25=bm25,
            multi_query=MultiQueryRetriever(
                hybrid, llm, query_count=cfg.multi_query_count, rrf_k=cfg.rrf_k, max_workers=max_workers
            ),
            rewriter=QueryRewriter(llm, use_hyde=use_hyde),
            grader=RelevanceGrader(llm if use_llm_grader else None),
            reranker=reranker,
            compressor=ContextCompressor(max_sentences=cfg.compression_max_sentences),
            expander=ParentExpander(document_store) if document_store is not None else None,
            document_store=document_store,
            enable_graph=enable_graph,
            knowledge_refresh=knowledge_refresh,
            mmr_lambda=mmr_lambda,
        )

    # -- corpus management ------------------------------------------------------------------------
    @property
    def corpus_size(self) -> int:
        with self._lock:
            return len(self._corpus)

    def refresh_corpus(self, chunks: Sequence[Chunk]) -> None:
        """Replace the lexical/graph corpus (parent chunks are excluded: they are context, not hits)."""
        with self._lock:
            self._corpus = {c.chunk_id: c for c in chunks if c.strategy is not ChunkStrategy.PARENT}
            self._rebuild()

    def add_to_corpus(self, chunks: Sequence[Chunk]) -> None:
        """Upsert chunks into the corpus by ``chunk_id``."""
        with self._lock:
            self._corpus.update({c.chunk_id: c for c in chunks if c.strategy is not ChunkStrategy.PARENT})
            self._rebuild()

    def _rebuild(self) -> None:
        chunks = list(self._corpus.values())
        self._bm25.refresh(chunks)
        self._graph = GraphRetriever(EntityGraph.build(chunks)) if self._enable_graph else None

    def _refresh_hook(self, company: str) -> KnowledgeRefresh | None:
        hook = self._knowledge_refresh
        if hook is None:
            return None
        store = self._store

        def refresh(query: str) -> int:
            ingested = hook(query)
            if ingested > 0 and store is not None:
                try:
                    self.add_to_corpus(store.list_chunks(company))
                except AgentError as exc:
                    _logger.warning("corpus_refresh_failed", error=str(exc)[:200])
            return ingested

        return refresh

    # -- retrieval --------------------------------------------------------------------------------
    @traced("retrieval.pipeline", span_type=SpanType.RETRIEVER)
    def retrieve(self, query: str, *, company: str, top_k: int | None = None) -> RetrievalOutcome:
        if not query.strip():
            raise ValueError("query must not be empty")
        started = time.perf_counter()
        k = top_k or self._settings.top_k
        pool = max(self._settings.candidate_pool, k)
        trace: list[str] = []

        rewrite = self._rewriter.rewrite(query, company=company)
        extras = [rewrite.rewritten, *([rewrite.hypothetical] if rewrite.hypothetical else [])]
        trace.append(f"rewrite:{rewrite.source} {rewrite.rewritten[:100]!r}")

        with self._lock:
            graph = self._graph if self._graph is not None and len(self._graph.graph) else None
        candidates = _CandidateSearch(self._multi_query, graph, extras, trace)
        crag = CorrectiveRetriever.from_settings(
            candidates,
            self._settings,
            grader=self._grader,
            rewriter=self._rewriter,
            knowledge_refresh=self._refresh_hook(company),
        )
        corrected = crag.retrieve(query, k=pool, company=company)
        trace.extend(f"crag:{line}" for line in corrected.action_trace)
        results = corrected.chunks

        if self._settings.rerank_enabled and results:
            results = self._reranker.rerank(query, results)
            trace.append(f"rerank:{type(self._reranker).__name__} n={len(results)}")

        results = maximal_marginal_relevance(results, k=k, lambda_mult=self._mmr_lambda)
        trace.append(f"mmr:selected={len(results)} lambda={self._mmr_lambda}")

        if self._expander is not None and results:
            results = self._expander.expand(results)
            attached = sum(1 for r in results if "parent_text" in r.chunk.metadata)
            trace.append(f"parent_expansion:attached={attached}")

        results = self._compressor.compress(query, results)[:k]
        trace.append(f"compression:returned={len(results)}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics = get_metrics()
        metrics.observe("retrieval.latency_ms", elapsed_ms)
        metrics.observe("retrieval.mean_relevance", corrected.final_relevance)
        metrics.increment("retrieval.requests")
        metrics.increment("retrieval.chunks_returned", len(results))
        _logger.info(
            "retrieval_complete",
            company=company,
            returned=len(results),
            verdict=corrected.verdict.value,
            mean_relevance=round(corrected.final_relevance, 3),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return RetrievalOutcome(
            query=query,
            company=company,
            chunks=results,
            trace=trace,
            corrective_actions=[a.value for a in corrected.actions],
            mean_relevance=corrected.final_relevance,
            verdict=corrected.verdict,
            rewritten_query=rewrite.rewritten,
            queries=list(dict.fromkeys([*corrected.queries, *candidates.queries])),
        )
