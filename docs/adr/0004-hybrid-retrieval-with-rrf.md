# ADR-0004: Hybrid retrieval (BM25 + dense) fused with weighted Reciprocal Rank Fusion

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

Qualification evidence mixes two query regimes:

- exact-token queries: tickers, fiscal periods, dollar amounts, product and
  programme names, executive titles ("CFO", "FY2026", "$4.2 billion");
- paraphrase queries: "cloud migration" versus "exited two data centres".

BM25 is strong on the first and weak on the second; dense retrieval is the
reverse. Their scores live on incomparable scales (unbounded BM25 versus cosine
similarity), so score-level blending needs calibration that drifts with the
corpus.

## Decision

- `retrieval/hybrid.py::HybridRetriever` runs `BM25Retriever` and
  `DenseRetriever` over a candidate pool of `retrieval.candidate_pool` (default
  40) and fuses the two ranked lists with
  `reciprocal_rank_fusion`, using only ranks: `weight / (rrf_k + rank)`.
  Defaults: `retrieval.rrf_k = 60`, `retrieval.dense_weight = 0.6` (BM25 gets
  `1 - dense_weight`).
- If the dense leg raises a dependency failure (endpoint down, breaker open),
  hybrid degrades to BM25 only and increments `retrieval.hybrid.dense_fallback`.
- RRF is reused at two other fusion points: `MultiQueryRetriever` fuses
  sub-query result lists (the original query weighted `ORIGINAL_QUERY_WEIGHT =
  1.5`), and `_CandidateSearch` in `retrieval/pipeline.py` fuses hybrid results
  with GraphRAG hits at `GRAPH_FUSION_WEIGHT = 0.5`.
- The BM25 corpus is an in-process snapshot of child chunks
  (`RetrievalPipeline.refresh_corpus`, parents excluded) swapped atomically so
  concurrent queries never see a half-built index.

## Consequences

- Positive: no score calibration; robust across corpora and embedding models.
- Positive: the retrieval quality gate in `tests/rag_eval/test_retrieval_quality.py`
  (recall@5 >= 0.8, MRR >= 0.6) asserts hybrid recall is never worse than the
  weaker single retriever.
- Positive: graceful degradation to lexical retrieval keeps briefs flowing during
  embedding outages.
- Negative: BM25 runs in the application process over the company's chunks
  loaded from `DocumentStore.list_chunks`. That is fine for per-company corpora
  (tens to low thousands of chunks) and would not scale to a global corpus.
  Mosaic AI Vector Search native hybrid (`DatabricksVectorIndex.hybrid_search`)
  is the migration path if per-request corpora grow.
- Negative: RRF discards score magnitude, so a single very strong dense hit
  receives no more credit than rank 1. Second-stage reranking (lexical features
  or LLM listwise) restores magnitude information.
