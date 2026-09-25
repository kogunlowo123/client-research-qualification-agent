# RAG Design

This document specifies the retrieval-augmented generation pipeline of the
Client Research & Qualification Agent: how public evidence is acquired, parsed,
chunked, enriched, embedded and stored, how it is retrieved for each
qualification criterion, and how generation is kept grounded. Every parameter
named here is a key in `src/client_research_agent/config/settings.py` or a
constant in the module cited next to it.

Related: [architecture.md](architecture.md), [threat_model.md](threat_model.md),
[ADR-0003](../adr/0003-delta-sync-index-self-managed-embeddings.md),
[ADR-0004](../adr/0004-hybrid-retrieval-with-rrf.md),
[ADR-0005](../adr/0005-deterministic-fallback-for-every-llm-step.md),
[ADR-0006](../adr/0006-verbatim-compression-for-verifiable-citations.md).

## 1. Design goals

1. **Every verified fact is traceable to a verbatim quote from a public page
   fetched in the same run.** This dominates every other trade-off.
2. Retrieval quality degrades gracefully, never catastrophically, when model
   endpoints fail (ADR-0005).
3. Identical behaviour locally and in Databricks: the pipeline is written
   against ports (`services/ports.py`), not SDKs.
4. Per-company corpora (tens to low thousands of chunks) are the unit of scale.
   The design is not a global web-scale search engine.

## 2. Pipeline overview

```text
 ingestion (research/)                         indexing (retrieval/indexing.py)
 ---------------------                         --------------------------------
 ResearchRequest                               SourceDocument[]
   |                                             |
   +-- EdgarClient (submissions, XBRL facts)     +-- ParentChildChunker.split
   +-- CorporateSiteDiscoverer (sitemaps, hubs)  |     parents: 1024 tok, no overlap
   +-- seed_urls / PublicAnalystSource           |     children: 256 tok, 32 overlap
   |                                             +-- MetadataEnricher.enrich
   v                                             |     contextual header, entities,
 PolicyEnforcingFetcher                          |     optional LLM situating sentence
   UrlGuard -> RobotsPolicy -> HostRateLimiter   +-- DocumentStore.save_documents
   -> retry + per-host CircuitBreaker            +-- DocumentStore.save_chunks
   |                                             |     parents -> parent_chunks
   v                                             +-- embed children (embedding_text)
 HtmlParser -> DocumentTypeClassifier            +-- VectorIndex.upsert -> chunks (+CDF)
 -> EntityExtractor -> SHA-256 dedupe                  -> Delta Sync chunks_index
   |
   v
 SourceDocument[] (trust-scored)


 retrieval (retrieval/pipeline.py::RetrievalPipeline.retrieve)
 -------------------------------------------------------------
 query, company
   -> QueryRewriter           (LLM rewrite [+ HyDE] | synonym expansion)
   -> MultiQueryRetriever     (LLM sub-queries | angle templates), parallel
        -> HybridRetriever    (BM25 + dense, weighted RRF)
   -> GraphRetriever          (entity co-occurrence expansion, RRF weight 0.5)
   -> CorrectiveRetriever     (grade; rewrite + re-retrieve; optional refresh)
   -> Reranker                (LexicalReranker | LLMReranker)
   -> maximal_marginal_relevance (lambda 0.7)
   -> ParentExpander          (attach parent_text, small-to-big)
   -> ContextCompressor       (verbatim sentence selection)
   -> top_k RetrievedChunk[] + trace[]
```

Every stage appends a line to `RetrievalOutcome.trace` (for example
`rewrite:llm ...`, `multi_query:llm queries=4 ...`, `crag:grade:initial mean=...`,
`mmr:selected=8 lambda=0.7`), which is recorded on MLflow spans and in the
audit record so reviewers can see exactly how evidence was obtained.

## 3. Ingestion

Module: `research/ingestion.py::IngestionPipeline.run(request) -> IngestionResult`.

| Step | Behaviour |
|---|---|
| Candidate collection | Concurrently from SEC EDGAR (recent filings and XBRL company facts), corporate-site discovery on `request.domain`, and `request.seed_urls`. Seeds on analyst domains are routed to `PublicAnalystSource` only ([ADR-0007](../adr/0007-public-sources-only-and-analyst-compliance.md)). |
| Crawl scope | The request's corporate domain and seed hosts are added to the allow-list for the duration of the run through the `crawl_scope` context variable, on top of `crawler.allowed_domain_suffixes` (default `["sec.gov"]`). The scope propagates into worker threads and never leaks across concurrent requests. |
| Fetch and parse | Bounded waves on a thread pool, stopping at `request.max_documents` (default 40, max 500). |
| Classification | `DocumentTypeClassifier`: weighted URL/title/body rules vote for a `DocumentType`; specific types beat the generic press-release bucket. |
| Entities | `EntityExtractor`: money, percentages, fiscal periods, headcount, executive appointments, technology terms (regex and gazetteer, no ML). |
| De-duplication | SHA-256 of normalised text (`content_hash`); hashes already present in the store (`DocumentStore.known_hashes`) are skipped. |
| Trust | `TRUST_SEC = 0.95`, `TRUST_COMPANY = 0.85`, `TRUST_ANALYST = 0.7`, `TRUST_OTHER = 0.5`. |
| Failure handling | A failing source or URL becomes a `SkippedSource` with a `SkipReason` (`policy_violation`, `http_status`, `fetch_failed`, `circuit_open`, `unsupported_content`, `empty_content`, `duplicate_content`, `already_ingested`, `max_documents_reached`, `source_failed`, `processing_error`) and never aborts the run. `IngestionResult.skipped_by_reason()` feeds the `ingestion.sources_skipped` metric. |

### Fetch policy

`research/fetcher.py::PolicyEnforcingFetcher` wraps the raw transport
`HttpxFetcher`:

- `UrlGuard.check` before every request and every redirect hop: http/https only,
  no credentials in the URL, default ports only, no IP literals or numeric host
  spellings, every resolved address must be globally routable, host must be in
  the allow-list.
- `RobotsPolicy` (RFC 9309): rules parsed from the first 500 KiB; 4xx means
  allow-all; 429, 5xx and network errors mean disallow-all (fail closed, cached
  for a shorter failure TTL); `Crawl-delay` slows the host through
  `HostRateLimiter.set_min_interval`.
- `HostRateLimiter`: per-host token bucket at
  `crawler.requests_per_second_per_host` (default 1.0; SEC fair access allows
  up to 10).
- Retry with backoff from `ResilienceSettings`, one `CircuitBreaker` per host.
- `HttpxFetcher`: streamed body capped at `crawler.max_response_bytes`
  (15,000,000; large 10-K primary documents exceed 5 MB) on the decompressed
  stream, content-type allow-list, 429 mapped to `RateLimitedError` honouring
  `Retry-After`, 5xx to `UpstreamServiceError`, timeouts to
  `UpstreamTimeoutError`.

## 4. Parsing

`research/parsing/html.py::HtmlParser.parse(html, url) -> ParsedPage`

- Boilerplate removal (scripts, styles, navigation, footers, forms, ARIA
  landmark roles such as `navigation`, `banner`, `contentinfo`).
- **Invisible-text removal**: elements with `hidden`, `aria-hidden="true"`,
  inline `display:none` / `visibility:hidden` / zero size, and screen-reader-only
  utility classes. Invisible text is the classic carrier of indirect prompt
  injection and retrieval poisoning, so it never reaches the index.
- Publication date resolution order: JSON-LD `datePublished` -> meta tags
  (`article:published_time`, `date`, `dc.date`, ...) -> `<time datetime>` ->
  dated text patterns near the top of the main content.

EDGAR JSON is parsed structurally in `research/sources/edgar.py`; XBRL facts
(`CompanyFacts`, `FactValue`, `latest_fact`) are reported values and feed the
company-scale revenue and headcount bands used by the heuristic qualifier.

## 5. Chunking

Module: `retrieval/chunking.py`. All chunkers operate on character **spans** of
the source text, recording `metadata["char_start"]` and `metadata["char_end"]`,
so chunk text is always an exact substring of the document. Token counts use the
`cl100k_base` BPE via `tiktoken` (`retrieval/tokenization.py`), degrading to
whitespace words when the BPE file cannot be loaded (air-gapped clusters).

| Strategy | Class | Parameters (settings key, default) | Use |
|---|---|---|---|
| Recursive | `RecursiveChunker` | `chunking.child_chunk_tokens` 256, `chunking.chunk_overlap_tokens` 32 | Paragraph -> line -> sentence -> word boundaries within a token budget. General purpose. |
| Semantic | `SemanticChunker` | `chunking.child_chunk_tokens` 256, `chunking.semantic_breakpoint_percentile` 90.0 | Embeds sentence windows (window 1) and breaks where cosine distance between neighbours exceeds the configured percentile; oversize groups are re-split recursively. Falls back to the recursive splitter on embedding failure (`metadata["semantic_fallback"] = true`). |
| Parent-child | `ParentChildChunker` | `chunking.parent_chunk_tokens` 1024, `chunking.child_chunk_tokens` 256, `chunking.chunk_overlap_tokens` 32 | **Default for indexing.** Parents are split without overlap; children are split within each parent with overlap and carry `parent_id`. Parents record `child_ids`. |

Validation in `ChunkingSettings`: `parent_chunk_tokens > child_chunk_tokens` and
`chunk_overlap_tokens < child_chunk_tokens`; child tokens are bounded to
[64, 2048], parent tokens to [256, 8192].

Chunk ids are deterministic (`make_chunk_id(doc_id, strategy, index)`), so
re-ingesting an unchanged document produces the same ids and the `MERGE`-based
writes are idempotent.

Why 256/1024: 256-token children are specific enough to score well against a
single-criterion query and short enough to quote; 1024-token parents give the
LLM the surrounding paragraph structure (tables, preceding sentences) that a
child alone lacks. The child stays the citation anchor.

## 6. Metadata enrichment and contextual retrieval

`retrieval/enrichment.py::MetadataEnricher`

- **Contextual header** (deterministic), prepended for embedding and lexical
  indexing through `Chunk.embedding_text`:
  `Document: <title> | Source: <document type> from <domain> | Published: <date> | Company: <company> | Section: <nearest heading>`.
  A chunk that says "revenue grew 12%" becomes findable for "Acme FY2026
  earnings" because its header carries that context.
- **Entities**: capitalised-run candidates (`extract_candidate_entities`, at most
  `MAX_ENTITIES_PER_CHUNK = 25`) stored on `Chunk.entities`; they build the
  GraphRAG entity graph.
- **Optional LLM situating sentence** (Anthropic-style contextual retrieval):
  bounded by `max_llm_chunks` (64 per call), `max_context_chars` (300) and
  `max_document_chars` (6000); any failure keeps the deterministic header and
  increments `enrichment.situating_fallback`.

## 7. Embeddings

| Model | Where | Dimension | Notes |
|---|---|---|---|
| `databricks-gte-large-en` | Production default (`serving.embedding_endpoint`) | 1024 (`serving.embedding_dimension`) | FMAPI pay-per-token; at most 150 inputs per request (`MAX_BATCH_SIZE`); vectors re-ordered by response `index` and dimension-checked before they reach the table. |
| `databricks-bge-large-en` | Alternative FMAPI endpoint | 1024 | Priced in the cost table (`observability/cost.py::DEFAULT_PRICING`) but not configured by default. |
| `HashingEmbeddingClient` (`cra-hashing-embedder-v1`) | Local and CI | 512 by default | Signed feature hashing of stemmed unigrams and bigrams, sublinear TF, L2-normalised. Deterministic and offline; lexical, not semantic. |

Trade-offs:

- **Quality.** GTE-large-en and BGE-large-en are comparable English retrieval
  models of the same size class and the same 1024-d output, so either fits the
  index without schema change. GTE-large-en is the default because it accepts a
  longer input context than BGE-large-en, which leaves headroom for a 256-token
  child plus its contextual header and optional situating sentence. The hashing
  embedder has no semantic generalisation; it exists to make retrieval quality
  gates reproducible in CI, not to approximate production quality.
- **Cost.** The default pricing table (an estimate to be overridden with contract
  rates) lists 1.857 DBU per million input tokens for GTE-large-en and 1.429 for
  BGE-large-en. Embedding cost is incurred once per child at ingestion because
  vectors are persisted in `chunks` (ADR-0003); query embeddings are one call per
  sub-query. At these rates embedding is a small fraction of per-brief cost; chat
  completion tokens dominate.
- **Latency.** Batching (`embed_in_batches`, batch size 64 in `IndexingPipeline`)
  and `CachingEmbeddingClient` (SHA-256-keyed LRU, 10,000 entries) remove
  duplicate work; the cache records `embedding.cache.hits` / `misses`.
- **Switching models** requires re-embedding all rows and recreating the index,
  because the index dimension is fixed at creation
  (`infrastructure/vector_search/index_spec.json`).

## 8. Storage

| Store | Content | Notes |
|---|---|---|
| UC `documents` | One row per fetched document, full text, `content_hash`, `trust_score` | CDF enabled; tag `data_classification = public`. |
| UC `chunks` | **Embedded child chunks only**, `embedding ARRAY<FLOAT> NOT NULL` | CDF enabled; CHECK `chunks_embedded_children_only`; Delta Sync source of `chunks_index`. |
| UC `parent_chunks` | Parent chunks, no embedding | Read by id for small-to-big expansion; CHECK `parent_chunks_parents_only`; never indexed. |
| Vector Search `chunks_index` | ANN index over `chunks.embedding` | Delta Sync, self-managed embeddings, `TRIGGERED`, endpoint `cra-vs-endpoint-<env>`. |
| In-process BM25 snapshot | Child chunks of the requested company | Rebuilt by `RetrievalPipeline.refresh_corpus` / `add_to_corpus`. |

All writes use `MERGE INTO` with the batch serialised as one JSON parameter and
expanded server-side (`inline(from_json(:rows, ...))`), so writes are
idempotent, atomic per batch and immune to SQL injection
(`databricks/unity_catalog.py`).

## 9. Retrieval

`RetrievalPipeline.retrieve(query, *, company, top_k=None)`. Defaults come from
`RetrievalSettings`.

| Stage | Implementation | Key parameters | Fallback |
|---|---|---|---|
| Query rewriting | `QueryRewriter.rewrite` | LLM keyword-rich rewrite | Company-name variants (legal suffixes stripped) plus a domain synonym table |
| HyDE | `QueryRewriter(use_hyde=True)` | Hypothetical answer passage added as an extra query | Skipped (`retrieval.hyde.fallback`) |
| Multi-query | `MultiQueryRetriever.retrieve_with_queries` | `retrieval.multi_query_count` 3 sub-queries, run in a thread pool, fused with RRF (`rrf_k`), original query weight 1.5 | Angle templates |
| BM25 | `BM25Retriever` | Okapi BM25 over `embedding_text` tokens; shared analyzer in `retrieval/lexical.py` | n/a |
| Dense | `DenseRetriever` | Query embedding + `VectorIndex.search` with `{"company": ...}` filter | Hybrid drops the dense leg |
| Hybrid | `HybridRetriever` | Pool `retrieval.candidate_pool` 40, `retrieval.rrf_k` 60, `retrieval.dense_weight` 0.6 | BM25 only (`retrieval.hybrid.dense_fallback`) |
| GraphRAG | `GraphRetriever` over `EntityGraph` | Seeds on entities named in the query, expands to neighbours; `k // 2` hits fused at weight 0.5 | Omitted when the graph is empty |
| CRAG | `CorrectiveRetriever` | Grades up to 8 chunks; mean relevance of the top 3 vs `retrieval.crag_min_relevance` 0.35; up to `retrieval.crag_max_corrections` 2 corrective rewrites; optional knowledge refresh | Lexical grader |
| Reranking | `LexicalReranker` (default) or `LLMReranker` | Lexical weights: coverage 0.30, BM25 0.25, prior 0.15, proximity 0.10, recency 0.10, trust 0.10; recency half-life `retrieval.recency_half_life_days` 365; enabled by `retrieval.rerank_enabled` | LLM path falls back to lexical |
| MMR | `maximal_marginal_relevance` | `lambda_mult` 0.7; cosine over embeddings when available, token Jaccard otherwise | n/a |
| Parent expansion | `ParentExpander` | Attaches `metadata["parent_text"]` (<= 8000 chars, centred on the child), de-duplicated per parent via `parent_text_ref` | Returns children unchanged |
| Compression | `ContextCompressor` | `retrieval.compression_max_sentences` 6, verbatim and in order; parent budget x2 | n/a (deterministic) |
| Top-k | `retrieval.top_k` 8 | | |

CRAG verdicts (`RetrievalVerdict`): `CORRECT` when mean relevance >= 0.35,
`AMBIGUOUS` when >= 0.175, `INCORRECT` otherwise. Irrelevant chunks are dropped
so they never reach the scorer as evidence. When the verdict is not `CORRECT`
and a `knowledge_refresh` hook is supplied, the pipeline invokes targeted
re-ingestion for the query, adds the new chunks to the corpus
(`add_to_corpus(store.list_chunks(company))`) and retrieves once more.

### Retrieval for qualification

`qualification/agent.py::QualificationAgent` issues, per criterion, the three
queries defined in `CriterionDefinition.queries` (with `{company}` substituted),
in parallel across criteria (`max_workers = 5`). Chunks for another company are
discarded. Candidates are then selected by `ranking/evidence_ranker.py::EvidenceRanker`:

```text
score = relevance_norm * SOURCE_TRUST[document_type] * recency(publication_date)
mmr   = 0.7 * score_norm - 0.3 * max_similarity_to_selected
```

with a soft per-domain cap of 3 so evidence is corroborated across sources.
`SOURCE_TRUST`: SEC filing 1.0, earnings release 0.95, investor relations 0.9,
press release 0.8, leadership announcement 0.75, analyst public 0.7, corporate
webpage 0.6. Undated documents get recency 0.5.

### Citation retrieval

Selected chunks are registered in the run's `EvidenceRegistry` in canonical
criterion order, receiving stable ids `E1..En` (de-duplicated by `chunk_id`).
Each `Evidence.quote` is the best contiguous supporting sentence window of the
chunk (<= 500 chars). Prompts render evidence as delimited blocks with
`sanitize_untrusted` applied (evidence-tag spoofing and control characters
neutralised). Every score, statement and citation check refers to evidence only
by these ids; an id the run did not register is rejected.

## 10. Generation and grounding

| Step | Module | Prompt template |
|---|---|---|
| Criterion scoring | `qualification/agent.py` | `criterion_qualifier` |
| Opportunity analysis | `briefing/opportunity.py` | `opportunity_analysis` |
| Narrative sections | `briefing/generator.py` | `brief_writer` |
| Discovery questions (exactly 5) | `briefing/generator.py` | `discovery_questions` |
| Citation judge | `citations/validator.py` | `citation_judge` |
| Research planning | `orchestration/steps.py::ResearchPlanner` | `research_planner` |

Templates are versioned Markdown with YAML front-matter
(`prompts/registry.py::PromptRegistry`). Rendering substitutes only declared
`{identifier}` placeholders in a single pass, so braces inside untrusted evidence
are never re-expanded. Each template's SHA-256 fingerprint is recorded in
`ClientBrief.model_versions`.

All structured outputs go through `services/structured.py::complete_structured`:
JSON mode, extraction from fenced blocks, Pydantic validation, bounded repair
turns with the validation error fed back, and `OutputValidationError` on failure.

### Citation validation

`citations/validator.py::CitationValidator.validate` (per fact statement):

1. Cited ids = `evidence_ids` plus inline `[E3]` markers (markers are stripped;
   the renderer re-adds them).
2. Ids unknown to the registry, with a non-http(s) URL, or whose URL was not
   retrieved in this run are rejected.
3. Support per id = `citations/entailment.py::lexical_support` (content-word
   coverage; any unmatched number caps support at 0.05; any missing entity caps
   it at 0.25), optionally blended with the LLM judge (weight 0.5). The judge is
   consulted only when all figures and entities are present, can lower support
   freely, and can raise it by at most `JUDGE_MAX_UPLIFT = 0.2`.
4. Support = max over cited ids. `>= guardrails.min_citation_support` (0.3; 0.35
   in prod): kept as `VERIFIED_FACT`. Unmatched figures: removed outright.
   Partial (`>= 0.5 * threshold`): re-labelled `AI_RECOMMENDATION` with an
   explicit inference note. Otherwise removed.

`CitationReport.coverage = supported_statements / total_statements` is the
share of claimed facts that survived grounding. It is the primary quality
metric for briefs and is alerted on (`citation_coverage_daily`).

## 11. Advanced RAG patterns

| Pattern | Status in this repository | Where |
|---|---|---|
| Corrective RAG (CRAG) | Implemented and on by default | `retrieval/crag.py`, stage of `RetrievalPipeline` |
| Self-RAG | Implemented as a critic (`Retrieve`, `IsRel`, `IsSup`, `IsUse` reflection tokens as validated JSON, token-overlap fallback) | `retrieval/self_rag.py::SelfRagCritic`; run by the orchestrator's `validation` step on the executive summary's verified facts (advisory, recorded in the step detail); not a stage of `RetrievalPipeline` |
| GraphRAG | Entity co-occurrence graph with label-propagation communities and extractive community summaries | `retrieval/graph.py`; fused into candidate search |
| Agentic RAG | Per-criterion retrieval plans (criterion queries, optional `research_planner` prompt), tool-like retrieval calls from the qualification agent, CRAG-driven knowledge refresh | `qualification/agent.py`, `retrieval/pipeline.py` |
| Citation RAG | Evidence ids in prompts, citation validation, verified-fact / recommendation separation | `qualification/evidence.py`, `citations/`, `briefing/` |
| Reflection-based retrieval | CRAG grading loop plus Self-RAG support critique | `retrieval/crag.py`, `retrieval/self_rag.py` |
| Contextual retrieval | Deterministic headers plus optional LLM situating sentence | `retrieval/enrichment.py` |
| Small-to-big | Parent-child chunking and `ParentExpander` | `retrieval/chunking.py`, `retrieval/parent_child.py` |
| Feedback loops | Reviewer feedback captured by the orchestration layer and turned into evaluation rows | `orchestration/review.py` (see [architecture.md](architecture.md#9-human-in-the-loop)) |
| Human-in-the-loop | Review queue for briefs before they are released | `orchestration/review.py` |

## 12. Evaluation metrics and gates

| Layer | Metric | Gate | Where enforced |
|---|---|---|---|
| Retrieval (offline, deterministic) | recall@5, MRR on a labelled corpus | recall@5 >= 0.8, MRR >= 0.6; hybrid recall >= min(BM25, dense) | `tests/rag_eval/test_retrieval_quality.py` (CI, `rag_eval` marker) |
| Grounding (offline) | Fabricated statements removed; grounded statements kept; every cited id resolves | Assertions per case | `tests/rag_eval/test_hallucination.py`, `tests/rag_eval/test_citation_verification.py` |
| Security (offline) | OWASP LLM Top 10 adversarial corpus | Assertions per case | `tests/security/test_owasp_llm_top10.py` |
| Per brief (online) | `CitationReport.coverage` | `--min-citation-coverage` = bundle var `eval_min_citation_coverage` (0.90), `--fail-on-gate` | `cra_brief_generation.evaluate` task |
| Dataset (nightly and pre-promotion) | Pass rate across MLflow GenAI judges; mean citation coverage | `eval_min_pass_rate` 0.85, `eval_min_citation_coverage` 0.90 | `cra_evaluation.evaluate_champion`, `cra_agent_deploy.evaluate_candidate` |
| Production | Mean daily citation coverage; not-enough-evidence share; guardrail blocks | 0.90; 0.40; 20/hour | `infrastructure/monitoring/alerts.sql` + `alerts.json` |

The gate itself is `evaluation/harness.py::QualityGate`, applied by
`EvaluationHarness`:

| Level | Criterion | Default |
|---|---|---|
| Per example | `citation_coverage` >= `min_citation_coverage` (or the example's own `expectations.min_citation_coverage`) | 0.9 |
| Per example | `grounded_fact_ratio` (facts whose best support reaches the threshold) >= `min_grounded_fact_ratio` | 0.8 |
| Per example | exactly five discovery questions; verdict equals the label when one is given | - |
| Per run | pass rate >= `min_pass_rate`; mean citation coverage >= `min_citation_coverage` | 0.85; 0.9 |
| Per run (optional) | `verdict_accuracy` >= `min_verdict_accuracy`; `criterion_score_mae` <= `max_criterion_mae` | off by default |

Metric functions live in `evaluation/metrics.py` (`citation_coverage`,
`grounded_fact_ratio`, `verdict_accuracy`, `criterion_score_mae`,
`section_completeness`, `discovery_questions_ok`, `expected_fact_recall`) and are
exposed to `mlflow.genai.evaluate` as scorers by `evaluation/mlflow_eval.py`.
Examples can carry captured document snapshots (`evaluation/snapshot.py`) so
they replay without network access.

The evaluation record schema (`eval_set` rows: `inputs`, `expectations` such as
`expected_verdict` and `expected_facts`) is defined in
`infrastructure/mlflow/eval_dataset_schema.json`.

## 13. Known limitations

- BM25 and the entity graph are built in process per company; they do not scale
  to cross-company search (ADR-0004).
- `TRIGGERED` index sync introduces a visibility delay between ingestion and
  dense retrieval (ADR-0003).
- The hashing embedder used in CI is lexical, so CI retrieval gates do not
  measure semantic recall of the production embedding model; the nightly
  dataset evaluation against the real endpoint does.
- Lexical entailment is conservative on paraphrase: a correct but heavily
  reworded fact can be downgraded to a recommendation. This is the intended
  failure direction.
